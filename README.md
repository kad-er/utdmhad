```mermaid
graph TD
    %% Style des blocs
    classDef skel fill:#4a47a3,stroke:#333,stroke-width:1px,color:#fff;
    classDef depth fill:#005f43,stroke:#333,stroke-width:1px,color:#fff;
    classDef imu fill:#7a3b1d,stroke:#333,stroke-width:1px,color:#fff;
    classDef fusion fill:#104f87,stroke:#333,stroke-width:1px,color:#fff;
    classDef head fill:#444,stroke:#333,stroke-width:1px,color:#fff;
    classDef out fill:#1a5fb4,stroke:#333,stroke-width:1px,color:#fff;

    %% Branche Skeleton
    S1["Skeleton<br>(T, 20 joints, 3 coords)"]:::skel --> S2["Conv1d × 2<br>60 → 64 → H ch.<br>+ AdaptPool → T=50"]:::skel
    S2 --> S3["LSTM<br>→ (B, 50, H)"]:::skel
    S3 --> S4["Temporal MHA<br>N heads, (B,T,H)→(B,H)"]:::skel

    %% Branche Depth
    D1["Depth video<br>(T, 1, 224, 224)"]:::depth --> D2["Conv2d × 3<br>1 → 32 → 64 → H ch.<br>+ AdaptAvgPool2d"]:::depth
    D2 --> D3["LSTM<br>→ (B, T, H)"]:::depth
    D3 --> D4["Temporal MHA<br>N heads, (B,T,H)→(B,H)"]:::depth

    %% Branche IMU
    I1["Inertial (IMU)<br>(T, 6 channels)"]:::imu --> I2["Conv1d × 2<br>6 → 64 → 256 ch.<br>+ BN + ReLU"]:::imu
    I2 --> I3["LSTM<br>→ (B, T, H)"]:::imu
    I3 --> I4["Temporal MHA<br>N heads, (B,T,H)→(B,H)"]:::imu

    %% Fusion
    S4 --> ST["stack → (B, 3, H)"]
    D4 --> ST
    I4 --> ST
    
    ST --> F1["Cross-modal fusion MHA<br>M heads over 3 modalities → (B,H)"]:::fusion
    
    %% Classification Head
    F1 --> H1["Linear(H,512) + BN + ReLU + Drop(0.3)"]:::head
    H1 --> H2["Linear(512,256) + BN + ReLU + Drop(0.2)"]:::head
    H2 --> O["Linear(256, 27)<br>27 action classes"]:::out
