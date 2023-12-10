import React, { useState, useRef, useEffect } from "react";
import Papa from "papaparse";
import './uploadPage.css';
import { useNavigate } from "react-router-dom";
import axios from 'axios';
import { FormData } from 'formdata-node';



const UploadPage = () => {
  const [data, setData] = useState([]); 
  const [columns, setColumns] = useState([]); 
  const [selected, setSelected] = useState(""); 
  const fileInput = useRef(null); 
  const [file, setFilePath] = useState(null);
  const [session_id, setSessionId] = useState(null);
  const [password, setPassword] = useState(null);
  const [selectedValue, setSelectedValue] = useState("");
  const [type, setType] = useState("");
  const [filepath, setFile, setFilepath] = useState(null);

  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setFile(file);
    // ------------------------------------Check---------------------------------
    Papa.parse(file, {
      header: true, 
      complete: (results) => {
        const { data, meta } = results;
        setData(results.data);
        setColumns(meta.fields);
        console.log(JSON.stringify(data));
      },
    });
    if (file){
      setFilePath(file.path);
      // console.log('File selected:', file);
      // setFilePath(URL.createObjectURL(file));
      // console.log(URL.createObjectURL(file))
      console.log(file.path);
    }
  };

  const handleSelect = (e) => {
    const value = e.target.value;
    setSelected(value);
  };

  const [isOpen, setIsOpen] = useState(false);

  const options = ["Regression", "Classification", "Clustering"];

  const [hashtag, setHashtag] = useState("");

  function handleToggle() {
    setIsOpen(!isOpen);
  }

  function handleSelected(option) {
    if(option === "Regression"){
      setSelectedValue("r");
      setType(option);

    }
    if(option === "Classification"){
      setSelectedValue("c");
      setType(option);
    }
    if(option === "Clustering"){
      setSelectedValue("cl");
      setType(option);
    }
    
    setIsOpen(false);
  }

  function getResults() {
    GetSessionIdAndPassword();
    handleSubmit();
    navigate('/results', {
      state: {
        session_id: session_id,
        password: password,
        target: selected,
        type: type,
      },
    });
  }

  const GetSessionIdAndPassword = async () => {
    // Define the config for the GET request
    const config = {
      method: "get",
      maxBodyLength: Infinity,
      url: "http://127.0.0.1:5000/session_id_gen",
      headers: {},
    };
  
    try {
      const response = await axios(config);
      setData(response.data);
      setSessionId(response.data.session_id);
      setPassword(response.data.password);
    } catch (error) {
      console.log(error);
    }
  };
  
  useEffect(() => {
    // Call the function here
    GetSessionIdAndPassword();
  }, []);
  

  const handleSubmit = async () => {

    const formData = new FormData();
    var axios = require('axios');
    var fs = require('fs');
    formData.append('session_id', session_id);
    formData.append('password', password);
    formData.append('option', selectedValue);
    formData.append('file', fs.createReadStream('/C:/Users/RST/Downloads/Telegram Desktop/Iris.csv'));

    var config = {
      method: 'post',
      maxBodyLength: Infinity,
      url: 'http://127.0.0.1:5000/upload',
      headers: {
        'Content-Type': 'multipart/form-data; boundary=' + formData._boundary
      },
      data: formData,
    };

    try {
      const response = await axios(config);
      console.log(JSON.stringify(response.data));
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="App">
        <div className="window">
            <div className="window2">
            <h2> SELECT ANALYSIS TYPE </h2>
        <div className="dropdown-menu">
      <button className="button-click" onClick={handleToggle}>
        {type || "Select an option"}
      </button>
      {isOpen && (
        <ul className="dropdown-list">
          {options.map((option) => (
            <li
              key={option}
              className="dropdown-item"
              onClick={() => handleSelected(option)}
            >
              {option}
            </li>
          ))}
        </ul>
      )}
      </div>

      <div className="button">
        <button className="button-click" onClick={getResults} > Analize Uploaded Data </button>
        <p>{hashtag}</p>
      </div>
      </div>
      <form>
      <input className="fileUpload"
        type="file"
        accept=".csv"
        ref={fileInput}
        onChange={handleFileChange}
        hidden
      />
      </form>
      </div>
      {data.length > 0 && (
        <div className="tableHeading">
            <div className="columnSelection">
            <h2>Column Selection to apply Analysis on</h2>
            {selected && <p>You have selected: {selected}</p>}
          <select value={selected} onChange={handleSelect}>
            <option value="">Select a column</option>
            {columns.map((column, index) => (
              <option key={index} value={column}>
                {column}
              </option>
            ))}
          </select>
          </div>
          <h2>Data Table</h2>
          <table>
            <thead>
              <tr>
                {columns.map((column, index) => (
                  <th key={index}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, index) => (
                <tr key={index}>
                  {columns.map((column, index) => (
                    <td key={index}>{row[column]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default UploadPage;
