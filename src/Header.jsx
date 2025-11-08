import React, { useState, useEffect } from "react";
import "./Header.css";
import leaf from "./assets/leaf.png";

function Header() {
  const [stats, setStats] = useState({ hectares: 0, trees: 0, co2: 0 });

  useEffect(() => {
    setStats({ hectares: 1234, trees: 56789, co2: 12 });
  }, []);

  return (
<header className="site-header">
  <div className="container header-inner">
    <a href="/" className="brand">
      <span className="brand-text">Leaf Watch</span>
      <img src={leaf} alt="" className="brand-icon" />
      </a>
      <section className="stats-card" aria-labelledby="stats-heading">
      <h2 id="stats-heading">🌳 Deforestation stats</h2>
      <ul>
        <li><strong>Hectares lost:</strong> <span className="value">{stats.hectares.toLocaleString()}</span></li>
        <li><strong>Trees cut:</strong>     <span className="value">{stats.trees.toLocaleString()}</span></li>
        <li><strong>CO₂ emitted:</strong>   <span className="value">{stats.co2} kt</span></li>
      </ul>
    </section>
  </div>
</header>
  );
}

export default Header;
