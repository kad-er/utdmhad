```mermaid
flowchart TD
    %% Définition des styles (couleurs de l'image originale)
    classDef skel fill:#3b3a7d,stroke:#555,stroke-width:1px,color:#fff;
    classDef depth fill:#004d38,stroke:#555,stroke-width:1px,color:#fff;
    classDef imu fill:#6e3319,stroke:#555,stroke-width:1px,color:#fff;
    classDef fusion fill:#0d47a1,stroke:#555,stroke-width:1px,color:#fff;
    classDef head fill:#424242,stroke:#555,stroke-width:1px,color:#fff;
    classDef baseline fill:#a1887f,stroke:#555,stroke-width:1px,color:#000;

    %% Configuration des colonnes parallèles en haut
    subgraph Colonnes [ ]
        direction LR
        style Colonnes fill:none,stroke:none;

        %% COLONNE 1 : SKELETON
        subgraph ColSkel [ ]
            direction TB
            style ColSkel fill:none,stroke:none;
            S1["Skeleton<br>(T, 20 joints, 3 coords)"]:::skel
            S2["Conv1d × 2<br>60 → 64 → H ch.<br>+ AdaptPool → T=50"]:::skel
            S3["LSTM<br>→ (B, 50, H)"]:::skel
            S4["Temporal MHA<br>N heads, (B,T,H)→(B,H)"]:::skel
            
            S1 --> S2 --> S3 --> S4
        end

        %% COLONNE 2 : DEPTH VIDEO
        subgraph ColDepth [ ]
            direction TB
            style ColDepth fill:none,stroke:none;
            D1["Depth video<br>(T, 1, 224, 224)"]:::depth
            D2["Conv2d × 3<br>1 → 32 → 64 → H ch.<br>+ AdaptAvgPool2d"]:::depth
            D3["LSTM<br>→ (B, T, H)"]:::depth
            D4["Temporal MHA<br>N heads, (B,T,H)→(B,H)"]:::depth
            
            D1 --> D2 --> D3 --> D4
        end

        %% COLONNE 3 : INERTIAL IMU
        subgraph ColImu [ ]
            direction TB
            style ColImu fill:none,stroke:none;
            I1["Inertial (IMU)<br>(T, 6 channels)"]:::imu
            I2["Conv1d × 2<br>6 → 64 → 256 ch.<br>+ BN + ReLU"]:::imu
            I3["LSTM<br>→ (B, T, H)"]:::imu
            I4["Temporal MHA<br>N heads, (B,T,H)→(B,H)"]:::imu
            
            I1 --> I2 --> I3 --> I4
        end
    end

    %% Section Cross-Modal et Fusion
    B1["Baseline variant: temporal MHA replaced by mean pooling over T"]:::baseline
    ST["stack → (B, 3, H)"]
    F1["Cross-modal fusion MHA<br>M heads over 3 modalities → (B,H)"]:::fusion
    B2["Baseline: concat(x1,x2,x3) → Linear(3H,H)"]:::baseline

    %% Connexions vers la fusion
    S4 --> B1
    D4 --> B1
    I4 --> B1
    B1 --> ST --> F1 --> B2

    %% Classification Head (En bas)
    H1["Linear(H, 512) + BN + ReLU + Drop(0.3)"]:::head
    H2["Linear(512, 256) + BN + ReLU + Drop(0.2)"]:::head
    O["Linear(256, 27)<br>27 action classes"]:::fusion

    B2 --> H1 --> H2 --> O
