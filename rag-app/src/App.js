import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState('');

  const handleSearch = async () => {
    try {
      const response = await axios.post('http://localhost:8000/generate', { query });
      setResult(response.data.result);
    } catch (error) {
      console.error("Error fetching the result:", error);
      setResult("An error occurred while fetching the result.");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAG Search Application</h1>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your search query"
        />
        <button onClick={handleSearch}>Search</button>
        <div className="result">
          <h2>Result:</h2>
          <p>{result}</p>
        </div>
      </header>
    </div>
  );
}

export default App;
