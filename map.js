// Initialize the map
const map = L.map('map').setView([20, 0], 2);

// Add OpenStreetMap tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
    minZoom: 2
}).addTo(map);

// Add click event to show coordinates
map.on('click', function(e) {
    const lat = e.latlng.lat.toFixed(4);
    const lng = e.latlng.lng.toFixed(4);
    
    L.popup()
        .setLatLng(e.latlng)
        .setContent(`<b>Location</b><br>Latitude: ${lat}<br>Longitude: ${lng}`)
        .openOn(map);
});

// Add some example markers for major cities
const cities = [
    { name: 'New York', coords: [40.7128, -74.0060] },
    { name: 'London', coords: [51.5074, -0.1278] },
    { name: 'Tokyo', coords: [35.6762, 139.6503] },
    { name: 'Sydney', coords: [-33.8688, 151.2093] },
    { name: 'Paris', coords: [48.8566, 2.3522] }
];

cities.forEach(city => {
    L.marker(city.coords)
        .addTo(map)
        .bindPopup(`<b>${city.name}</b>`);
});