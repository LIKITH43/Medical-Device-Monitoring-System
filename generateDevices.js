function generateDevices() {
    devices = [];
    const deviceNames = Object.keys(DEVICE_MAPPING);
    
    for (let i = 0; i < 24; i++) {
        const deviceName = deviceNames[i];
        const deviceType = DEVICE_MAPPING[deviceName];
        const location = LOCATIONS[Math.floor(Math.random() * LOCATIONS.length)];
        
        const device = {
            id: i + 1,
            deviceName,
            deviceType,
            location,
            alerted: false,
            lastUpdate: new Date(),
            isUpdating: false,
            ...generateRealisticParameters(deviceType)
        };
        
        devices.push(device);
    }
    
    console.log('Generated 24 medical devices with realistic parameters');
}
