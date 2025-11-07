import React, { useState, useEffect } from "react";
import "./Header.css";
import leaf from "./assets/leaf.png";

function Header() {
  // ← this is the right place
  const [stats, setStats] = useState({ hectares: 0, trees: 0, co2: 0 });

  // (optional) load/update the numbers later
  useEffect(() => {
    // Example: replace with a real fetch
    setStats({ hectares: 1234, trees: 56789, co2: 12 });
  }, []);

  return (
    <header className="site-header">
      <a href="/" className="brand">
        <span className="brand-text">Leaf Watch</span>
        <img src={leaf} alt="" className="brand-icon" />
      </a>

      <section className="stats-card" aria-labelledby="stats-heading">
        <h2 id="stats-heading">Deforestation stats</h2>
        <ul>
          <li><strong>Hectares lost:</strong> {stats.hectares.toLocaleString()}</li>
          <li><strong>Trees cut:</strong> {stats.trees.toLocaleString()}</li>
          <li><strong>CO₂ emitted:</strong> {stats.co2} kt</li>
        </ul>
      </section>
    </header>
  );
}

export default Header;
