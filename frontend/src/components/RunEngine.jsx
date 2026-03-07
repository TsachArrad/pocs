import { useState } from 'react'
import axios from 'axios'
import './RunEngine.css'

function RunEngine() {
  const [fileName, setFileName] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)

  const handleRun = async (e) => {
    e.preventDefault()
    if (!fileName.trim()) return

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const res = await axios.get(`/api/run-engine/${encodeURIComponent(fileName)}`)
      setResponse(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="run-engine-container">
      <h2>Run Connector Engine</h2>
      <p className="description">
        Execute a connector using engine.py in a Docker container. The connector will be fetched from storage, validated, and executed.
      </p>

      <form onSubmit={handleRun} className="run-engine-form">
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
          {loading ? 'Running...' : 'Run Connector'}
        </button>
      </form>

      {error && (
        <div className="error-box">
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && (
        <div className={response.ok ? 'success-box' : 'error-box'}>
          <h3>{response.ok ? '✓ Success' : '✗ Failed'}</h3>
          <div className="result-details">
            {response.exit_code !== undefined && (
              <p><strong>Exit Code:</strong> {response.exit_code}</p>
            )}
            {response.duration_ms !== undefined && (
              <p><strong>Duration:</strong> {response.duration_ms}ms</p>
            )}
            {response.error && (
              <div className="error-details">
                <strong>Error:</strong>
                <pre>{response.error}</pre>
              </div>
            )}
            {response.result !== undefined && (
              <div className="result-data">
                <strong>Result:</strong>
                <pre>{JSON.stringify(response.result, null, 2)}</pre>
              </div>
            )}
            {response.stdout && (
              <div className="output-section">
                <strong>Standard Output:</strong>
                <pre>{response.stdout}</pre>
              </div>
            )}
            {response.stderr && (
              <div className="output-section stderr">
                <strong>Standard Error:</strong>
                <pre>{response.stderr}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default RunEngine
