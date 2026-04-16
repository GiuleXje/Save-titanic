```mermaid
graph LR
    %% Alimentare
    USB[USB-C Connector] -->|5V| BQ[BQ25180 LiPo Charger]
    BQ -->|Incarcare| BAT[LiPo Battery 3.7V]
    BAT --> MAX[MAX17048 Fuel Gauge]
    BAT --> RT[RT6160 DC/DC Converter]
    
    %% Conexiuni Power catre MCU
    RT -->|3.3V| MCU[nRF52840 MCU + BLE]

    %% I2C Bus
    BQ <-->|I2C| MCU
    MAX <-->|I2C| MCU
    RT <-->|I2C| MCU
    IMU[BMA421 IMU] <-->|I2C| MCU
    HAPTIC[DRV2605 Haptic Driver] <-->|I2C| MCU

    %% Alte interfete
    MCU -->|SPI| EPD[E-paper Display]
    MCU ---|RF| ANT[Antena BLE]
    BTN[Butoane Tactile] -->|GPIO| MCU
    SWD[TC2030 SWD Debug] <-->|SWD| MCU
