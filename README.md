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


Componentă	Descriere	Cod JLC	Link JLC Parts	Datasheet
nRF52840-QIAA-R	MCU Bluetooth 5.4, ARM Cortex-M4F	C190794	Link JLC	Link
BQ25180YBGR	LiPo Charger & Power Path Management	C3682423	Link JLC	Link
RT6160AWSC	Buck-Boost DC/DC Converter 3.3V	C7065276	Link JLC	Link
MAX17048G+T10	Fuel Gauge (Monitorizare baterie)	C2682616	Link JLC	Link
BMA421	Accelerometru 3 axe (Pedometer embedded)	C5242966	Link JLC	Link
DRV2605LDGSR	Haptic Driver (Control vibrații)	C527464	Link JLC	Link
LCM1027B3605F	Motor vibrații ERM (Shaker)	C7528806	Link JLC	Link
SI2301CDS	P-Channel MOSFET (EPD Power Gating)	C10487	Link JLC	Link
GDEH0154D67	1.54" E-paper Display (200x200)	-	[Manual Order]*	Link
AKYGA LP502030	Baterie LiPo 250mAh (3.7V)	-	[Manual Order]*	Link
