import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './results.css';

function Results({ session_id, password, target, type }) {

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Use useEffect hook to fetch the data when the component mounts or the props change
  useEffect(() => {

    // Create a cancel token source
    const source = axios.CancelToken.source();

    const fetchData = async () => {

      setLoading(true);
      setError(null);

      const data = `{"session_id":${session_id},"password":${password},"target":${target}}`;
      const config = {
        method: "get",
        maxBodyLength: Infinity,
        url: `http://127.0.0.1:5000/${type}`,  // classifiction in privious version
        headers: {},
        data: data,
        // Pass the cancel token to the config
        cancelToken: source.token
      };
  
      try {
        const response = await axios(config);
        setData(response.data);
        setLoading(false);
      } catch (error) {
        // Check if the error is caused by cancellation
        if (axios.isCancel(error)) {
          console.log("Request cancelled");
        } else {
          console.log(error);
        }
      }
    };
    fetchData();

    // Return a cleanup function that cancels the request
    return () => {
      source.cancel();
    };
  }, [session_id, password, target]);
  

  return (
    <div className="Results">
      <h1>Results</h1>
      {loading ? (
        <div className="loader"></div>
      ) : error ? (
        <div className="error">Error: {error}</div>
      ) : data ? (
        <table>
          <thead>
            <tr>
              {Object.keys(data[0]).map((key) => (
                <th key={key}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={index}>
                {Object.values(item).map((value) => (
                  <td key={value}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        null
      )}
    </div>
  );
}

export default Results;
