import { useState } from 'react'
import axios from 'axios'
import './Validation.css'

function Validation() {
  const [fileName, setFileName] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)

  const handleValidate = async (e) => {
    e.preventDefault()
    if (!fileName.trim()) return

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const res = await axios.get(`/api/validate-connector/${encodeURIComponent(fileName)}`)
      setResponse(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="validation-container">
      <h2>Validate Connector</h2>
      <p className="description">
        Validate a connector JSON from storage using engine.py validators. Enter the filename to fetch and validate.
      </p>

      <form onSubmit={handleValidate} className="validation-form">
        <div className="form-group">
          <label htmlFor="fileName">File Name</label>
          <input
            id="fileName"
            type="text"
            value={fileName}
            onChange={(e) => setFileName(e.target.value)}
            placeholder="e.g., mondaykata.json"
            disabled={loading}
          />
          <small className="helper-text">
            The connector will be fetched from: https://uploaderbe-b4dbh9eec3hmh5ep.westeurope-01.azurewebsites.net/api/Connector/get-file/[filename]
          </small>
        </div>

        <button type="submit" disabled={loading || !fileName.trim()} className="btn-primary">
          {loading ? 'Validating...' : 'Validate Connector'}
        </button>
      </form>

      {error && (
        <div className="error-box">
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && (
        <div className={response.valid ? 'success-box' : 'error-box'}>
          <h3>{response.valid ? '✓ Valid' : '✗ Invalid'}</h3>
          <div className="result-details">
            <p><strong>File:</strong> {response.file_name}</p>
            {response.message && <p><strong>Message:</strong> {response.message}</p>}
            {response.error && (
              <div className="error-details">
                <strong>Validation Error:</strong>
                <pre>{response.error}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default Validation
