# 🏏 Fantasy Cricket Team Selector

A machine learning-powered Streamlit app that predicts and selects the optimal fantasy cricket team using neural networks.

## Features

- **AI-Powered Predictions**: Uses scikit-learn MLPRegressor to predict fantasy scores
- **Smart Team Selection**: Automatically selects players based on roles (Wicketkeeper, Batsman, Allrounder, Bowler)
- **Captain & Vice-Captain**: Identifies top performers for captain and vice-captain roles
- **CSV Upload Support**: Load your cricket player data directly

## Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/DineshKumar0715/gamathon.git
cd gamathon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
```

### Run with Docker

```bash
docker build -t gamathon .
docker run -p 8501:8501 gamathon
```

## Deployment

### Streamlit Cloud

1. Push to GitHub: `https://github.com/DineshKumar0715/gamathon`
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select this repository
4. Set main file to `app/main.py`
5. Click Deploy!

## Usage

1. Open the app in your browser
2. Upload your cricket dataset CSV file (with fantasy points)
3. (Optional) Upload additional player data
4. The app will train a neural network and suggest the optimal 11-player team
5. View predicted scores, captain, and vice-captain selections

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- streamlit
- fastapi
- uvicorn

## Data Format

Your CSV should include these columns:
- `Player`: Player name
- `Total_Runs`: Total runs scored
- `Balls_Faced`: Balls faced
- `Times_Out`: Times dismissed
- `4s`: Number of 4s
- `6s`: Number of 6s
- `Strike_Rate`: Strike rate
- `Wickets`: Wickets taken
- `Balls_Bowled`: Balls bowled
- `Economy`: Economy rate
- `Fantasy_Batting`: Batting fantasy points
- `Fantasy_Bowling`: Bowling fantasy points
- `Total_Fantasy_Points`: Total fantasy points

## Live Demo

🔗 [Deployed App](https://gamathon.streamlit.app)

## Author

DineshKumar0715

## License

MIT
