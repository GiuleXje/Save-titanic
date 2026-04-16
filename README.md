```mermaid
graph TD
    %% Definește grupuri funcționale
    subgraph Power[Power Management]
        USB_C[USB-C Input] -->|5V| Charger[MCP73832 Charger]
        Charger -->|Charge| Battery[LiPo 250mAh Battery\n32.5x21x5.5mm]
        Battery -->|Power| Regulator[3.3V Regulator]
        Regulator -->|3.3V| Rail[Power Rail]
    end

    subgraph Processing[Processing Unit]
        MCU[nRF52840 MCU\nARM Cortex-M4F, BLE]
    end

    subgraph Sensing[Sensing Layer]
        PM_Sensor[PMSA003 PM Sensor\nPM1.0, PM2.5, PM10]
        CO2_Sensor[MH-Z19B CO2 Sensor\nCO2 NDIR]
        Env_Sensor[BME680 Env Sensor\nVOC, Temp, Humidity, Pressure]
        Buttons[Buttons\nGPIO x3]
    end

    subgraph Output[Output/HMI]
        Display[1.54" E-Ink Display\nSPI, 200x200px]
        Shaker[Haptic Shaker\nERM Motor]
    end

    %% Conexiuni de Alimentare (Linii întrerupte)
    Rail -.->|3.3V| MCU
    Rail -.->|3.3V| PM_Sensor
    Rail -.->|3.3V| CO2_Sensor
    Rail -.->|3.3V| Env_Sensor
    Rail -.->|3.3V| Display
    Rail -.->|3.3V| Shaker

    %% Conexiuni de Date (Linii solide)
    MCU -->|SPI| Display
    MCU ---|I2C shared bus| Env_Sensor
    MCU ---|I2C shared bus| Shaker
    MCU <-->|UART serial| PM_Sensor
    MCU <-->|UART serial| CO2_Sensor
    MCU <-->|GPIO| Buttons
