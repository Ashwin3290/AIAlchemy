import "./HeroSection.css"
import { useNavigate } from "react-router-dom"; // import useNavigate hook

const HeroSection = () => {
    const navigate = useNavigate(); // create navigate variable
    return (
      <div className="HeroSection">
        <div id="box1"></div>
        <div id="box2">
            <div id="text">
                LETS START WITH
            </div>
            <div id="text">
                DATA ANALYSIS
            </div>

            <button id="getStarted" onClick={() => navigate("/upload")}> Get started </button>

        </div>
        <div id="box3">
            <div id="container">
                <div id="logo">
                    <img src=""/>
                </div>
                <div id="menu">
                    <ul>
                        <li onClick={() => navigate("/upload")}>Get Started</li>
                        <li>Our Team</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    )
  }

  export default HeroSection
