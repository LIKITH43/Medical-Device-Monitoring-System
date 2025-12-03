function generateDevices() {
    devices = [];
    const deviceNames = Object.keys(DEVICE_MAPPING);

    const DEVICES_PER_LOCATION = 5; // tweak this number to scale up/down
    let idCounter = 1;

    LOCATIONS.forEach((location, locIndex) => {
        for (let i = 0; i < DEVICES_PER_LOCATION; i++) {
            // Cycle through the known device names
            const deviceName = deviceNames[(idCounter - 1) % deviceNames.length];
            const deviceType = DEVICE_MAPPING[deviceName];

            const device = {
                id: idCounter,
                deviceName,
                deviceType,
                location,
                alerted: false,
                lastUpdate: new Date(),
                isUpdating: false,
                ...generateRealisticParameters(deviceType)
            };

            devices.push(device);
            idCounter++;
        }
    });

    console.log(`Generated ${devices.length} medical devices across ${LOCATIONS.length} locations`);
}
