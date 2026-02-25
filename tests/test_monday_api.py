# monday_smart_create_item.py
#
# Purpose:
#   - Create a monday.com item on a board by BOARD NAME
#   - For each provided column title -> string value:
#       * If column is free-text: send string directly
#       * If column is UI-option based (status/dropdown/people): fetch allowed options/users and match
#
# ENV:
#   MONDAY_API_TOKEN   (required)
#   BOARD_NAME         (required)
#   ITEM_NAME          (optional; default "Connectivity test")
#   COLUMN_VALUES_JSON (optional; JSON object: {"Column Title": "string value", ...})

import os
import json
import requests

MONDAY_API_URL = "https://api.monday.com/v2"

FREE_TEXT_TYPES = {
    "text",
    "long_text",
    "numbers",
    "email",
    "phone",
    "link",
}


def gql(token: str, query: str, variables: dict | None = None) -> dict:
    headers = {"Authorization": token, "Content-Type": "application/json"}
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(MONDAY_API_URL, headers=headers, json=payload, timeout=30)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    data = r.json()
    if data.get("errors"):
        raise RuntimeError(f"GraphQL errors: {json.dumps(data['errors'], ensure_ascii=False)}")
    return data["data"]


def find_board_id_by_name(token: str, board_name: str, limit: int = 200) -> tuple[int, str]:
    q = """
    query ($limit: Int!) {
      boards(limit: $limit) { id name }
    }
    """
    d = gql(token, q, {"limit": limit})
    for b in d.get("boards", []):
        if (b.get("name") or "").strip().lower() == board_name.strip().lower():
            return int(b["id"]), b.get("name") or board_name
    raise RuntimeError(f"Board not found by name (searched first {limit}): {board_name}")


def get_board_columns(token: str, board_id: int) -> tuple[str, dict]:
    q = """
    query ($id: [ID!]) {
      boards(ids: $id) {
        id
        name
        columns { id title type settings }
      }
    }
    """
    d = gql(token, q, {"id": [str(board_id)]})
    boards = d.get("boards") or []
    if not boards:
        raise RuntimeError(f"Board id not found: {board_id}")
    b = boards[0]
    cols = b.get("columns") or []
    by_title = {(c.get("title") or "").strip().lower(): c for c in cols}
    return b.get("name") or "", by_title


def get_users(token: str) -> list[dict]:
    q = """
    query {
      users { id name email }
    }
    """
    d = gql(token, q)
    out = []
    for u in d.get("users") or []:
        out.append(
            {
                "id": int(u["id"]),
                "name": (u.get("name") or "").strip(),
                "email": (u.get("email") or "").strip().lower(),
            }
        )
    return out


def extract_labels_from_settings(settings_obj) -> list[str]:
    # settings is JSON (settings_str is deprecated)
    if not isinstance(settings_obj, dict):
        return []

    labels: list[str] = []
    if isinstance(settings_obj.get("labels"), dict):
        for _, v in settings_obj["labels"].items():
            if isinstance(v, str) and v.strip():
                labels.append(v.strip())
    elif isinstance(settings_obj.get("labels"), list):
        for v in settings_obj["labels"]:
            if isinstance(v, str) and v.strip():
                labels.append(v.strip())

    # unique while preserving order
    seen = set()
    uniq = []
    for x in labels:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def match_label(allowed_labels: list[str], desired_value: str) -> str | None:
    desired = (desired_value or "").strip().lower()

    # exact
    for lbl in allowed_labels:
        if lbl.strip().lower() == desired:
            return lbl

    # contains (soft)
    for lbl in allowed_labels:
        if desired and desired in lbl.strip().lower():
            return lbl

    return None


def encode_status(label: str) -> dict:
    # Common accepted format
    return {"label": label}


def encode_dropdown(label: str) -> dict:
    # Common accepted format (single select); you can extend to multi-select by splitting input.
    return {"labels": [label]}


def encode_people(user_id: int) -> dict:
    # Common accepted format
    return {"personsAndTeams": [{"id": int(user_id), "kind": "person"}]}


def create_item(token: str, board_id: int, item_name: str, column_values_by_id: dict) -> tuple[int, str]:
    m = """
    mutation ($board: ID!, $name: String!, $colVals: JSON!) {
      create_item(board_id: $board, item_name: $name, column_values: $colVals) {
        id
        name
      }
    }
    """
    variables = {
        "board": str(board_id),
        "name": item_name,
        # monday expects a JSON-encoded string for column_values
        "colVals": json.dumps(column_values_by_id, ensure_ascii=False),
    }
    d = gql(token, m, variables)
    item = d["create_item"]
    return int(item["id"]), item.get("name") or item_name


def main():
    
    token = "eyJhbGciOiJIUzI1NiJ9.eyJ0aWQiOjYyMjc3ODI2NywiYWFpIjoxMSwidWlkIjoxMDAwMTA1ODUsImlhZCI6IjIwMjYtMDItMThUMTM6Mzk6NDQuMDAwWiIsInBlciI6Im1lOndyaXRlIiwiYWN0aWQiOjMzODU4MjcxLCJyZ24iOiJldWMxIn0.C_Ilyo09-kR5FApX2Evrpk6fAwvusPCVhJTmJcaDYnM"
    board_name = "Tasks"
    item_name = "Sprint1"
    col_values_json =  '{"Name": "Task 1", "Owner": "tzach112@gmail.com", "Status": "Done", "Type": "Feature", "TaskId": "T-1234" ,"Rstimated SP": "5","Epic": "Epic 1","Githun Link": "dasdas"}'


    if not token:
        raise RuntimeError("Missing MONDAY_API_TOKEN")
    if not board_name:
        raise RuntimeError("Missing BOARD_NAME")

    values_by_title = json.loads(col_values_json)
    if not isinstance(values_by_title, dict):
        raise RuntimeError("COLUMN_VALUES_JSON must be a JSON object/dict")

    # Validate input is string->string
    for k, v in values_by_title.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise RuntimeError("All keys/values in COLUMN_VALUES_JSON must be strings")

    board_id, resolved_board_name = find_board_id_by_name(token, board_name)
    _, cols_by_title = get_board_columns(token, board_id)

    # Fetch users only if needed (people column present)
    need_users = False
    for title in values_by_title.keys():
        col = cols_by_title.get(title.strip().lower())
        if col and col.get("type") == "people":
            need_users = True
            break
    users = get_users(token) if need_users else []

    column_values_by_id = {}
    applied = []

    for title, raw_value in values_by_title.items():
        col = cols_by_title.get(title.strip().lower())
        if not col:
            raise RuntimeError(f"Column not found on board '{resolved_board_name}': '{title}'")

        col_id = col.get("id")
        col_type = col.get("type")
        settings = col.get("settings")

        matched_value = None
        encoded_value = None

        if col_type in FREE_TEXT_TYPES:
            encoded_value = raw_value
            matched_value = raw_value

        elif col_type == "status":
            allowed = extract_labels_from_settings(settings)
            matched = match_label(allowed, raw_value)
            if not matched:
                raise RuntimeError(
                    f"No match for status '{raw_value}' in column '{title}'. Allowed: {allowed}"
                )
            encoded_value = encode_status(matched)
            matched_value = matched

        elif col_type == "dropdown":
            allowed = extract_labels_from_settings(settings)
            matched = match_label(allowed, raw_value)
            if not matched:
                raise RuntimeError(
                    f"No match for dropdown '{raw_value}' in column '{title}'. Allowed: {allowed}"
                )
            encoded_value = encode_dropdown(matched)
            matched_value = matched

        elif col_type == "people":
            desired = raw_value.strip().lower()

            chosen = None
            # Prefer email
            for u in users:
                if u["email"] and u["email"] == desired:
                    chosen = u
                    break
            # Fallback name
            if not chosen:
                for u in users:
                    if u["name"].strip().lower() == desired:
                        chosen = u
                        break

            if not chosen:
                raise RuntimeError(
                    f"No match for people '{raw_value}' in column '{title}'. Provide user email or exact name."
                )

            encoded_value = encode_people(chosen["id"])
            matched_value = chosen["email"] or chosen["name"]

        else:
            # Fallback: try raw string (works for some column types)
            encoded_value = raw_value
            matched_value = raw_value

        column_values_by_id[col_id] = encoded_value
        applied.append(
            {
                "title": title,
                "type": col_type,
                "inputValue": raw_value,
                "matchedValue": matched_value,
            }
        )

    item_id, created_name = create_item(token, board_id, item_name, column_values_by_id)

    print(
        json.dumps(
            {
                "ok": True,
                "boardId": board_id,
                "boardName": resolved_board_name,
                "itemId": item_id,
                "itemName": created_name,
                "applied": applied,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
