# Project Flow Diagram

```mermaid
flowchart TD
    A[User Access Home Page] --> B{User?}
    B -->|New User| C[Signup with Email & Password]
    C --> D[Send OTP Email]
    D --> E[Verify OTP]
    E -->|Success| F[Create User in DB]
    F --> G[Login]
    B -->|Existing User| G[Login]
    G --> H{Admin or User?}
    H -->|Admin| I[Admin Dashboard]
    H -->|User| J[User Dashboard]

    J --> K[Upload Dataset (CSV)]
    K --> L[Validate & Save Dataset]
    L --> M[Load Dataset for Visualization]

    M --> N[Visualize Data]
    N --> O[Filter by Region]
    O --> P[Seasonal Decomposition & Analysis]
    P --> Q[Gradient Boosting Forecasting]
    Q --> R[Generate Charts & Metrics]
    R --> S[Display Visualization]

    I --> T[Manage Users (Delete etc.)]

    subgraph Data Generation
        U[data_gen.py Script]
        U --> V[Generate Synthetic Data]
        V --> W[Save to dataset.csv]
    end

    S -->|Logout| A
    I -->|Logout| A
    J -->|Logout| A
