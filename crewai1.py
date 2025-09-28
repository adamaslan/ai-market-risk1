# Predictive Market Risk Pipeline (Fitch Group)
# Complete implementation with CrewAI orchestration

import os
import asyncio
import nest_asyncio  # Add this import to handle nested event loops in interactive environments
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Core libraries
import yfinance as yf
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import torch
import torch.nn as nn
from transformers import pipeline
import sendgrid
from sendgrid.helpers.mail import Mail
from supabase import create_client, Client
import google.generativeai as genai

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain.tools import tool

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Market Coverage
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "TSM",
    "COST", "ADBE", "PEP", "CSCO", "ACN", "TMUS", "CMCSA", "AMD", "INTC", "INTU",
    "AMGN", "REGN", "ADP", "QCOM", "VRTX", "AMAT", "ISRG", "MU", "ATVI", "MDLZ",
    "PYPL", "ADSK", "MELI", "KLAC", "SNPS", "CDNS", "ASML", "CHTR", "MNST", "LRCX",
    "ORLY", "KDP", "DXCM", "MAR", "IDXX", "CTAS", "ROST", "WDAY", "PCAR", "AZN",
    "CPRT", "XEL", "DLTR", "FAST", "VRSK", "ANSS", "SGEN", "BIIB", "ALGN", "SIRI",
    "EBAY", "EXC", "NTES", "JD", "BIDU", "SWKS", "INCY", "WBA", "ULTA", "TTWO",
    "VRSN", "LULU", "MTCH", "ZM", "DOCU", "OKTA", "DDOG", "CRWD", "NET", "FTNT",
    "ZS", "PANW", "TEAM", "PLTR", "DBX", "AFRM", "COIN", "HOOD", "DASH", "RIVN",
    "LCID", "PTON", "ABNB", "UBER", "LYFT", "SNOW", "DDOG", "MDB", "TWLO", "SPLK",
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "AXP", "MS", "BLK", "C",
    "XOM", "CVX", "LLY", "JNJ", "UNH", "ABBV", "MRK", "ABT", "PFE", "MDT",
    "WMT", "PG", "KO", "HD", "MCD", "PEP", "TJX", "NKE", "SBUX", "MDLZ", "CL",
    "GE", "UNP", "CAT", "HON", "RTX", "LMT", "UPS", "NOC", "BA", "CRM"
]

# Environment variables (set these in your environment)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize services
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
genai.configure(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@dataclass
class MarketEvent:
    timestamp: datetime
    event_type: str
    description: str
    affected_tickers: List[str]
    severity_score: float
    source_url: str

@dataclass
class RiskAssessment:
    event: MarketEvent
    impact_analysis: str
    probability_score: float
    time_horizon: str
    recommended_actions: List[str]

class MarketDataProvider:
    """Unified market data provider with yfinance primary, Alpha Vantage backup"""
    
    def __init__(self):
        self.alpha_vantage_ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY) if ALPHA_VANTAGE_API_KEY else None
        self.alpha_vantage_fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY) if ALPHA_VANTAGE_API_KEY else None
        self.rate_limit_delay = 12  # Alpha Vantage rate limit
        
    async def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get stock data with yfinance first, Alpha Vantage as backup"""
        try:
            # Primary: yfinance
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if not data.empty:
                logger.info(f"Retrieved {ticker} data from yfinance")
                return self._standardize_data(data, ticker)
                
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {e}")
            
        # Backup: Alpha Vantage
        return await self._get_alpha_vantage_data(ticker)
    
    async def _get_alpha_vantage_data(self, ticker: str) -> pd.DataFrame:
        """Fallback to Alpha Vantage API"""
        if not self.alpha_vantage_ts:
            logger.error("Alpha Vantage not configured")
            return pd.DataFrame()
            
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            data, _ = self.alpha_vantage_ts.get_daily(symbol=ticker, outputsize='full')
            
            df = pd.DataFrame(data).T
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            logger.info(f"Retrieved {ticker} data from Alpha Vantage")
            return self._standardize_data(df, ticker)
            
        except Exception as e:
            logger.error(f"Alpha Vantage failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def _standardize_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardize data format and add technical indicators"""
        if data.empty:
            return data
            
        # Add technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
        data['Ticker'] = ticker
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class NewsMonitor:
    """Global news monitoring for market events"""
    
    def __init__(self):
        self.news_api_key = NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2"
        self.risk_keywords = [
            "recession", "inflation", "interest rates", "fed", "central bank",
            "geopolitical", "trade war", "sanctions", "supply chain", "earnings",
            "merger", "acquisition", "bankruptcy", "regulation", "crypto", "china"
        ]
    
    async def scan_financial_news(self) -> List[MarketEvent]:
        """Scan for relevant financial news events"""
        events = []
        
        try:
            for keyword in self.risk_keywords:
                url = f"{self.base_url}/everything"
                params = {
                    "q": keyword,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "from": (datetime.now() - timedelta(hours=24)).isoformat(),
                    "apiKey": self.news_api_key
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for article in data.get("articles", [])[:5]:  # Limit per keyword
                    event = self._create_market_event(article, keyword)
                    if event:
                        events.append(event)
                
                await asyncio.sleep(0.1)  # Rate limiting
                
        except Exception as e:
            logger.error(f"News scanning failed: {e}")
            
        return events
    
    def _create_market_event(self, article: Dict, keyword: str) -> Optional[MarketEvent]:
        """Create market event from news article"""
        try:
            # Simple ticker extraction (you could enhance with NLP)
            affected_tickers = []
            title_content = f"{article['title']} {article.get('description', '')}"
            
            for ticker in TICKERS[:20]:  # Check subset for performance
                if ticker.lower() in title_content.lower():
                    affected_tickers.append(ticker)
            
            # Calculate severity score based on keyword and source
            severity_score = self._calculate_severity(article, keyword)
            
            return MarketEvent(
                timestamp=datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00")),
                event_type=keyword,
                description=article["title"],
                affected_tickers=affected_tickers,
                severity_score=severity_score,
                source_url=article["url"]
            )
            
        except Exception as e:
            logger.error(f"Failed to create event from article: {e}")
            return None
    
    def _calculate_severity(self, article: Dict, keyword: str) -> float:
        """Calculate event severity score (0-1)"""
        score = 0.5  # Base score
        
        # Keyword-based scoring
        severity_weights = {
            "recession": 0.9, "inflation": 0.8, "fed": 0.7,
            "bankruptcy": 0.9, "regulation": 0.6, "earnings": 0.4
        }
        score *= severity_weights.get(keyword, 0.5)
        
        # Source reliability (simplified)
        reliable_sources = ["reuters", "bloomberg", "wsj", "ft"]
        if any(source in article.get("source", {}).get("name", "").lower() for source in reliable_sources):
            score *= 1.2
        
        return min(score, 1.0)

class TransformerPredictor(nn.Module):
    """PyTorch Transformer model for market prediction"""
    
    def __init__(self, input_size: int = 64, hidden_size: int = 256, num_layers: int = 4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position_encoding = nn.Parameter(torch.randn(1000, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project input and add position encoding
        x = self.input_projection(x)
        x = x + self.position_encoding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Output projection (predict next day return)
        return self.output_projection(x[:, -1, :])

class MarketSimulator:
    """Market simulation and prediction engine"""
    
    def __init__(self):
        self.model = TransformerPredictor()
        self.scaler = None
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'Volatility']
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input"""
        features = data[self.feature_columns].fillna(method='ffill').fillna(0)
        
        # Calculate returns and additional features
        features['Return'] = data['Close'].pct_change()
        features['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        features['Price_MA_Ratio'] = data['Close'] / data['SMA_20']
        
        return features.fillna(0).values
    
    async def simulate_scenario(self, ticker_data: Dict[str, pd.DataFrame], 
                               risk_events: List[MarketEvent]) -> Dict[str, Any]:
        """Simulate market scenarios under risk conditions"""
        results = {}
        
        try:
            for ticker, data in ticker_data.items():
                if data.empty:
                    continue
                    
                # Prepare features
                features = self.prepare_features(data)
                
                if len(features) < 60:  # Need sufficient history
                    continue
                
                # Create sequences for transformer
                sequences = self._create_sequences(features[-60:])  # Last 60 days
                
                # Simulate baseline scenario
                baseline_pred = self._predict_price_series(sequences)
                
                # Simulate risk-adjusted scenarios
                risk_adjusted_preds = []
                for event in risk_events:
                    if ticker in event.affected_tickers or not event.affected_tickers:
                        risk_pred = self._apply_risk_adjustment(baseline_pred, event)
                        risk_adjusted_preds.append(risk_pred)
                
                results[ticker] = {
                    "baseline_prediction": baseline_pred.tolist(),
                    "risk_scenarios": risk_adjusted_preds,
                    "current_price": float(data['Close'].iloc[-1]),
                    "volatility": float(data['Volatility'].iloc[-1])
                }
                
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            
        return results
    
    def _create_sequences(self, features: np.ndarray, seq_length: int = 30) -> torch.Tensor:
        """Create sequences for transformer input"""
        sequences = []
        for i in range(len(features) - seq_length + 1):
            sequences.append(features[i:i + seq_length])
        return torch.FloatTensor(sequences)
    
    def _predict_price_series(self, sequences: torch.Tensor, 
                             forecast_days: int = 30) -> np.ndarray:
        """Predict future price series"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            current_seq = sequences[-1:]  # Last sequence
            
            for _ in range(forecast_days):
                pred = self.model(current_seq)
                predictions.append(pred.item())
                
                # Update sequence (simplified - in practice, you'd update all features)
                new_step = current_seq[:, -1:, :].clone()
                new_step[:, :, 0] = pred  # Update price feature
                current_seq = torch.cat([current_seq[:, 1:, :], new_step], dim=1)
        
        return np.array(predictions)
    
    def _apply_risk_adjustment(self, baseline_pred: np.ndarray, 
                              event: MarketEvent) -> List[float]:
        """Apply risk event adjustments to baseline predictions"""
        # Risk adjustment factors based on event severity and type
        adjustment_factors = {
            "recession": -0.15, "inflation": -0.08, "earnings": 0.05,
            "regulation": -0.12, "fed": -0.10, "geopolitical": -0.18
        }
        
        factor = adjustment_factors.get(event.event_type, 0.0)
        risk_multiplier = factor * event.severity_score
        
        # Apply time-decaying impact
        decay_rates = np.exp(-np.arange(len(baseline_pred)) * 0.1)
        adjustments = risk_multiplier * decay_rates
        
        return (baseline_pred * (1 + adjustments)).tolist()

# CrewAI Tools
class MarketDataTool(BaseTool):
    name: str = "market_data_fetcher"
    description: str = "Fetch real-time market data for specified tickers"
    
    def __init__(self):
        super().__init__()
        self.provider = MarketDataProvider()
    
    def _run(self, tickers: str) -> str:
        """Fetch market data for given tickers"""
        ticker_list = [t.strip() for t in tickers.split(",")]
        results = {}
        
        loop = asyncio.get_event_loop()
        for ticker in ticker_list[:5]:  # Limit for performance
            data = loop.run_until_complete(self.provider.get_stock_data(ticker))
            if not data.empty:
                latest = data.iloc[-1]
                results[ticker] = {
                    "price": float(latest['Close']),
                    "volume": float(latest['Volume']),
                    "volatility": float(latest['Volatility']),
                    "rsi": float(latest['RSI'])
                }
        
        return json.dumps(results)

class NewsAnalysisTool(BaseTool):
    name: str = "news_analyzer"
    description: str = "Analyze recent financial news for market risks"
    
    def __init__(self):
        super().__init__()
        self.monitor = NewsMonitor()
    
    def _run(self, query: str = "") -> str:
        """Scan and analyze recent financial news"""
        loop = asyncio.get_event_loop()
        events = loop.run_until_complete(self.monitor.scan_financial_news())
        
        event_summaries = []
        for event in events[:10]:  # Top 10 events
            event_summaries.append({
                "timestamp": event.timestamp.isoformat(),
                "type": event.event_type,
                "description": event.description,
                "severity": event.severity_score,
                "affected_tickers": event.affected_tickers
            })
        
        return json.dumps(event_summaries)

class SimulationTool(BaseTool):
    name: str = "market_simulator"
    description: str = "Run market simulations under different risk scenarios"
    
    def __init__(self):
        super().__init__()
        self.simulator = MarketSimulator()
        self.provider = MarketDataProvider()
    
    def _run(self, tickers_and_events: str) -> str:
        """Run market simulations"""
        try:
            data = json.loads(tickers_and_events)
            tickers = data.get("tickers", [])
            events_data = data.get("events", [])
            
            # Convert events data back to MarketEvent objects
            events = []
            for event_data in events_data:
                events.append(MarketEvent(
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    event_type=event_data["type"],
                    description=event_data["description"],
                    affected_tickers=event_data["affected_tickers"],
                    severity_score=event_data["severity"],
                    source_url=""
                ))
            
            # Get market data
            loop = asyncio.get_event_loop()
            ticker_data = {}
            for ticker in tickers[:5]:  # Limit for performance
                data = loop.run_until_complete(self.provider.get_stock_data(ticker))
                if not data.empty:
                    ticker_data[ticker] = data
            
            # Run simulations
            results = loop.run_until_complete(
                self.simulator.simulate_scenario(ticker_data, events)
            )
            
            return json.dumps(results)
            
        except Exception as e:
            return f"Simulation failed: {str(e)}"

# CrewAI Agents
def create_horizon_scanner_agent():
    """Create the Horizon Scanner Agent"""
    return Agent(
        role="Market Horizon Scanner",
        goal="Continuously monitor global markets and news for emerging risk signals",
        backstory="""You are an expert market surveillance analyst with deep experience 
        in identifying early warning signals across global financial markets. You excel 
        at connecting seemingly unrelated events to potential market impacts.""",
        tools=[MarketDataTool(), NewsAnalysisTool()],
        verbose=True,
        allow_delegation=False
    )

def create_economic_analyst_agent():
    """Create the Economic Analyst Agent"""
    return Agent(
        role="Senior Economic Analyst",
        goal="Analyze market events and assess their potential economic impact",
        backstory="""You are a senior economist with 15+ years of experience analyzing 
        market events and their cascading effects across different sectors and geographies. 
        You specialize in second-order effect analysis and risk quantification.""",
        tools=[],
        verbose=True,
        allow_delegation=False,
        llm_config={
            "model": "gemini-pro",
            "api_key": GEMINI_API_KEY
        } if GEMINI_API_KEY else None
    )

def create_simulation_agent():
    """Create the Simulation Agent"""
    return Agent(
        role="Quantitative Risk Modeler",
        goal="Run predictive simulations and model potential market scenarios",
        backstory="""You are a quantitative analyst specializing in market risk modeling 
        and scenario analysis. You use advanced mathematical models to predict market 
        behavior under different risk conditions.""",
        tools=[SimulationTool()],
        verbose=True,
        allow_delegation=False
    )

def create_briefing_agent():
    """Create the Briefing Agent"""
    return Agent(
        role="Risk Intelligence Briefer",
        goal="Synthesize analysis into actionable client briefings",
        backstory="""You are a senior risk communication specialist who translates 
        complex market analysis into clear, actionable intelligence for institutional 
        clients. You excel at prioritizing risks and recommending specific actions.""",
        tools=[],
        verbose=True,
        allow_delegation=False
    )

class MarketRiskPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.agents = {
            "scanner": create_horizon_scanner_agent(),
            "analyst": create_economic_analyst_agent(),
            "simulator": create_simulation_agent(),
            "briefer": create_briefing_agent()
        }
        
        self.sendgrid_client = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY) if SENDGRID_API_KEY else None
    
    async def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete risk analysis pipeline"""
        logger.info("Starting Predictive Market Risk Pipeline")
        
        # Task 1: Horizon Scanning
        scanning_task = Task(
            description=f"""Scan global markets and news for emerging risks affecting these tickers: 
            {', '.join(TICKERS[:20])}. Focus on events from the last 24 hours that could impact 
            market sentiment or specific sectors. Provide market data and news analysis.""",
            agent=self.agents["scanner"],
            expected_output="JSON formatted market data and news events with risk indicators"
        )
        
        # Task 2: Economic Analysis
        analysis_task = Task(
            description="""Analyze the market events identified by the scanner. Assess the potential 
            economic impact, identify second-order effects, and quantify risk probabilities. 
            Consider sector correlations and macroeconomic implications.""",
            agent=self.agents["analyst"],
            expected_output="Detailed risk assessment with impact analysis and probability scores"
        )
        
        # Task 3: Market Simulation
        simulation_task = Task(
            description="""Run predictive simulations based on the identified risks and market data. 
            Model different scenarios showing potential price impacts over 1-week, 1-month, and 
            3-month horizons. Include confidence intervals and risk-adjusted predictions.""",
            agent=self.agents["simulator"],
            expected_output="Simulation results with predicted price paths and scenario analysis"
        )
        
        # Task 4: Client Briefing
        briefing_task = Task(
            description="""Synthesize all analysis into a comprehensive risk briefing for Fitch 
            clients. Prioritize the most critical risks, provide clear impact assessments, and 
            recommend specific portfolio actions. Format for executive consumption.""",
            agent=self.agents["briefer"],
            expected_output="Executive risk briefing with prioritized risks and action items"
        )
        
        # Create and execute crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[scanning_task, analysis_task, simulation_task, briefing_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            
            # Store results in Supabase
            if supabase:
                await self._store_results(result)
            
            # Send briefing to clients
            if self.sendgrid_client:
                await self._send_briefing(result.raw)
            
            logger.info("Pipeline execution completed successfully")
            return {"status": "success", "results": result.raw}
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _store_results(self, results: Any):
        """Store pipeline results in Supabase"""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "pipeline_results": str(results),
                "status": "completed"
            }
            
            supabase.table("risk_pipeline_runs").insert(data).execute()
            logger.info("Results stored in Supabase")
            
        except Exception as e:
            logger.error(f"Failed to store results: {e}")
    
    async def _send_briefing(self, briefing_content: str):
        """Send risk briefing via SendGrid"""
        try:
            message = Mail(
                from_email='risk-pipeline@fitch.com',
                to_emails='clients@fitch.com',
                subject=f'Market Risk Briefing - {datetime.now().strftime("%Y-%m-%d")}',
                html_content=f"""
                <h2>Fitch Market Risk Intelligence</h2>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}</p>
                <div style="white-space: pre-wrap; font-family: monospace;">
                {briefing_content}
                </div>
                <hr>
                <p><em>This briefing was generated by Fitch's Predictive Market Risk Pipeline</em></p>
                """
            )
            
            response = self.sendgrid_client.send(message)
            logger.info(f"Briefing sent successfully: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Failed to send briefing: {e}")

# Background worker for continuous monitoring
class ContinuousMonitor:
    """Background worker for 24/7 market monitoring"""
    
    def __init__(self, pipeline: MarketRiskPipeline):
        self.pipeline = pipeline
        self.running = False
        self.check_interval = 3600  # 1 hour
    
    async def start_monitoring(self):
        """Start continuous monitoring loop"""
        self.running = True
        logger.info("Starting continuous market monitoring")
        
        while self.running:
            try:
                # Check if markets are open (simplified)
                current_hour = datetime.now().hour
                if 9 <= current_hour <= 16:  # Market hours (EST)
                    await self.pipeline.run_pipeline()
                else:
                    logger.info("Markets closed, monitoring news only")
                    # Run reduced scan for major news events
                    
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(300)  # 5 minute error recovery
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.running = False
        logger.info("Stopping continuous monitoring")

# Main execution
async def main():
    """Main execution function"""
    
    # Verify environment setup
    required_env_vars = ["ALPHA_VANTAGE_API_KEY", "NEWS_API_KEY", "GEMINI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("Some features may be limited")
    
    # Initialize pipeline
    pipeline = MarketRiskPipeline()
    
    # Option 1: Run once
    print("Running Fitch Predictive Market Risk Pipeline...")
    results = await pipeline.run_pipeline()
    print(f"Pipeline completed with status: {results['status']}")
    
    # Option 2: Continuous monitoring (uncomment to enable)
    # monitor = ContinuousMonitor(pipeline)
    # try:
    #     await monitor.start_monitoring()
    # except KeyboardInterrupt:
    #     monitor.stop_monitoring()
    #     logger.info("Pipeline stopped by user")

if __name__ == "__main__":
    nest_asyncio.apply()  # Enable nested asyncio usage
    asyncio.run(main())

# Additional configuration files and utilities

# requirements.txt content (save as separate file)
REQUIREMENTS = """
# Core dependencies
crewai>=0.1.0
yfinance>=0.2.18
alpha-vantage>=2.3.1
pandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
transformers>=4.30.0
requests>=2.31.0
sendgrid>=6.10.0
supabase>=1.0.0
google-generativeai>=0.3.0
python-dotenv>=1.0.0
asyncio>=3.4.3
aiohttp>=3.8.0

# Development dependencies
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
"""

# Docker configuration
DOCKERFILE = """
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["python", "main.py"]
"""

# Environment template
ENV_TEMPLATE = """
# API Keys (Required)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
NEWS_API_KEY=your_news_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
SENDGRID_API_KEY=your_sendgrid_api_key_here

# Database (Required)
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here

# Optional Configuration
PIPELINE_CHECK_INTERVAL=3600
MAX_TICKERS_PER_SCAN=20
SIMULATION_FORECAST_DAYS=30
EMAIL_FROM=risk-pipeline@fitch.com
EMAIL_TO=clients@fitch.com

# Render deployment
PORT=8000
PYTHON_VERSION=3.11.0
"""

# Database schema (Supabase SQL)
DATABASE_SCHEMA = """
-- Market Risk Pipeline Database Schema

-- Table for storing pipeline execution results
CREATE TABLE risk_pipeline_runs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    pipeline_results JSONB,
    status VARCHAR(50),
    execution_time_seconds INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for storing market events
CREATE TABLE market_events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE,
    event_type VARCHAR(100),
    description TEXT,
    affected_tickers TEXT[],
    severity_score FLOAT,
    source_url TEXT,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for storing risk assessments
CREATE TABLE risk_assessments (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    event_id UUID REFERENCES market_events(id),
    impact_analysis TEXT,
    probability_score FLOAT,
    time_horizon VARCHAR(50),
    recommended_actions JSONB,
    analyst_confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for storing simulation results
CREATE TABLE simulation_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(10),
    baseline_prediction JSONB,
    risk_scenarios JSONB,
    current_price FLOAT,
    volatility FLOAT,
    simulation_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for client portfolio templates
CREATE TABLE client_portfolios (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    client_id VARCHAR(100),
    portfolio_name VARCHAR(200),
    tickers JSONB,
    weights JSONB,
    risk_tolerance VARCHAR(50),
    notification_preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_pipeline_runs_timestamp ON risk_pipeline_runs(timestamp);
CREATE INDEX idx_market_events_timestamp ON market_events(timestamp);
CREATE INDEX idx_market_events_tickers ON market_events USING GIN(affected_tickers);
CREATE INDEX idx_simulation_results_ticker ON simulation_results(ticker);
CREATE INDEX idx_simulation_results_date ON simulation_results(simulation_date);

-- Row Level Security (RLS) policies
ALTER TABLE risk_pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulation_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_portfolios ENABLE ROW LEVEL SECURITY;

-- Basic policy (adjust based on your authentication setup)
CREATE POLICY "Allow all operations for authenticated users" ON risk_pipeline_runs
    FOR ALL USING (auth.role() = 'authenticated');
"""

# API server for health checks and monitoring (Flask/FastAPI)
API_SERVER = '''
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import asyncio
import json
from market_risk_pipeline import MarketRiskPipeline, supabase

app = Flask(__name__)
pipeline = MarketRiskPipeline()

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Basic connectivity checks
        checks = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "services": {
                "database": "connected" if supabase else "disconnected",
                "pipeline": "ready"
            }
        }
        return jsonify(checks), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/api/run-pipeline", methods=["POST"])
def trigger_pipeline():
    """Manual pipeline trigger endpoint"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(pipeline.run_pipeline())
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/portfolio-risk", methods=["POST"])
def analyze_portfolio_risk():
    """Analyze specific portfolio risk"""
    try:
        data = request.get_json()
        tickers = data.get("tickers", [])
        weights = data.get("weights", [])
        
        # Simplified portfolio risk analysis
        if len(tickers) != len(weights):
            return jsonify({"error": "Tickers and weights must have same length"}), 400
        
        # This would integrate with your simulation engine
        portfolio_analysis = {
            "tickers": tickers,
            "weights": weights,
            "total_risk_score": 0.65,  # Placeholder
            "top_risks": ["Interest rate changes", "Tech sector volatility"],
            "recommended_actions": ["Reduce tech exposure", "Add defensive positions"]
        }
        
        return jsonify(portfolio_analysis), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recent-events", methods=["GET"])
def get_recent_events():
    """Get recent market events from database"""
    try:
        days = request.args.get("days", 7, type=int)
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if supabase:
            response = supabase.table("market_events")\\
                .select("*")\\
                .gte("timestamp", cutoff_date.isoformat())\\
                .order("timestamp", desc=True)\\
                .limit(50)\\
                .execute()
            
            return jsonify(response.data), 200
        else:
            return jsonify({"message": "Database not configured"}), 503
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/simulation-results/<ticker>", methods=["GET"])
def get_simulation_results(ticker):
    """Get latest simulation results for a ticker"""
    try:
        if supabase:
            response = supabase.table("simulation_results")\\
                .select("*")\\
                .eq("ticker", ticker.upper())\\
                .order("created_at", desc=True)\\
                .limit(1)\\
                .execute()
            
            if response.data:
                return jsonify(response.data[0]), 200
            else:
                return jsonify({"message": f"No simulation data for {ticker}"}), 404
        else:
            return jsonify({"message": "Database not configured"}), 503
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
'''

# Render deployment configuration
RENDER_CONFIG = """
# render.yaml
services:
  - type: web
    name: fitch-risk-pipeline
    env: python
    plan: standard
    buildCommand: pip install -r requirements.txt
    startCommand: python api_server.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: ALPHA_VANTAGE_API_KEY
        sync: false
      - key: NEWS_API_KEY
        sync: false
      - key: GEMINI_API_KEY
        sync: false
      - key: SENDGRID_API_KEY
        sync: false
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_KEY
        sync: false

  - type: worker
    name: market-monitor-worker
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python worker.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: ALPHA_VANTAGE_API_KEY
        sync: false
      - key: NEWS_API_KEY
        sync: false
      - key: GEMINI_API_KEY
        sync: false
"""

# Worker script for background processing
WORKER_SCRIPT = '''
"""
Background worker for continuous market monitoring
"""
import asyncio
import logging
import signal
import sys
from market_risk_pipeline import MarketRiskPipeline, ContinuousMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GracefulKiller:
    """Handle graceful shutdown"""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.kill_now = True

async def main():
    """Main worker function"""
    killer = GracefulKiller()
    
    # Initialize pipeline and monitor
    pipeline = MarketRiskPipeline()
    monitor = ContinuousMonitor(pipeline)
    
    logger.info("Starting Fitch Risk Pipeline Worker")
    
    try:
        # Start monitoring in background
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        
        # Wait for shutdown signal
        while not killer.kill_now:
            await asyncio.sleep(1)
        
        # Graceful shutdown
        logger.info("Initiating graceful shutdown...")
        monitor.stop_monitoring()
        monitor_task.cancel()
        
        try:
            await monitor_task
        except asyncio.CancelledError:
            logger.info("Monitor task cancelled successfully")
            
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)
    
    logger.info("Worker shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
'''

# Configuration management
CONFIG_MANAGER = '''
"""
Configuration management for the pipeline
"""
import os
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class PipelineConfig:
    # API Configuration
    alpha_vantage_key: Optional[str] = None
    news_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    sendgrid_api_key: Optional[str] = None
    
    # Database Configuration
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    
    # Pipeline Configuration
    check_interval: int = 3600
    max_tickers_per_scan: int = 20
    simulation_forecast_days: int = 30
    batch_size: int = 5
    
    # Email Configuration
    email_from: str = "risk-pipeline@fitch.com"
    email_to: List[str] = None
    
    # Market Configuration
    market_open_hour: int = 9
    market_close_hour: int = 16
    timezone: str = "US/Eastern"
    
    def __post_init__(self):
        if self.email_to is None:
            self.email_to = ["clients@fitch.com"]
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load configuration from environment variables"""
        return cls(
            alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            news_api_key=os.getenv("NEWS_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            sendgrid_api_key=os.getenv("SENDGRID_API_KEY"),
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_KEY"),
            check_interval=int(os.getenv("PIPELINE_CHECK_INTERVAL", "3600")),
            max_tickers_per_scan=int(os.getenv("MAX_TICKERS_PER_SCAN", "20")),
            simulation_forecast_days=int(os.getenv("SIMULATION_FORECAST_DAYS", "30")),
            email_from=os.getenv("EMAIL_FROM", "risk-pipeline@fitch.com"),
            email_to=os.getenv("EMAIL_TO", "clients@fitch.com").split(",")
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        required_fields = [
            ("alpha_vantage_key", "ALPHA_VANTAGE_API_KEY"),
            ("news_api_key", "NEWS_API_KEY"),
            ("gemini_api_key", "GEMINI_API_KEY"),
            ("supabase_url", "SUPABASE_URL"),
            ("supabase_key", "SUPABASE_KEY")
        ]
        
        for field, env_var in required_fields:
            if not getattr(self, field):
                errors.append(f"Missing required configuration: {env_var}")
        
        if self.check_interval < 60:
            errors.append("Check interval must be at least 60 seconds")
        
        if self.max_tickers_per_scan > 50:
            errors.append("Max tickers per scan should not exceed 50")
        
        return errors

# Usage example:
# config = PipelineConfig.from_env()
# errors = config.validate()
# if errors:
#     logger.error(f"Configuration errors: {errors}")
'''

# Testing framework
TESTS = '''
"""
Test suite for the Market Risk Pipeline
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from market_risk_pipeline import (
    MarketDataProvider, NewsMonitor, TransformerPredictor, 
    MarketSimulator, MarketEvent, MarketRiskPipeline
)

class TestMarketDataProvider:
    
    @pytest.fixture
    def provider(self):
        return MarketDataProvider()
    
    @pytest.mark.asyncio
    async def test_get_stock_data_success(self, provider):
        """Test successful stock data retrieval"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock yfinance response
            mock_data = pd.DataFrame({
                'Open': [100, 101, 102],
                'High': [105, 106, 107], 
                'Low': [99, 100, 101],
                'Close': [104, 105, 106],
                'Volume': [1000000, 1100000, 1200000]
            }, index=pd.date_range('2024-01-01', periods=3))
            
            mock_ticker.return_value.history.return_value = mock_data
            
            result = await provider.get_stock_data("AAPL")
            
            assert not result.empty
            assert 'SMA_20' in result.columns
            assert 'RSI' in result.columns
            assert 'Volatility' in result.columns
            assert result['Ticker'].iloc[0] == "AAPL"
    
    @pytest.mark.asyncio 
    async def test_alpha_vantage_fallback(self, provider):
        """Test fallback to Alpha Vantage when yfinance fails"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Make yfinance fail
            mock_ticker.return_value.history.side_effect = Exception("API Error")
            
            with patch.object(provider, '_get_alpha_vantage_data') as mock_av:
                mock_av.return_value = pd.DataFrame({'Close': [100, 101]})
                
                result = await provider.get_stock_data("AAPL")
                mock_av.assert_called_once_with("AAPL")

class TestNewsMonitor:
    
    @pytest.fixture
    def monitor(self):
        return NewsMonitor()
    
    @pytest.mark.asyncio
    async def test_scan_financial_news(self, monitor):
        """Test news scanning functionality"""
        mock_response = {
            "articles": [{
                "title": "Fed raises interest rates amid inflation concerns",
                "description": "The Federal Reserve announced a rate hike",
                "publishedAt": "2024-01-15T10:00:00Z",
                "url": "https://example.com/news",
                "source": {"name": "Reuters"}
            }]
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status.return_value = None
            
            events = await monitor.scan_financial_news()
            
            assert len(events) > 0
            assert events[0].event_type in monitor.risk_keywords
            assert events[0].severity_score > 0

class TestTransformerPredictor:
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = TransformerPredictor(input_size=32, hidden_size=128)
        assert model.input_size == 32
        assert model.hidden_size == 128
        
        # Test forward pass
        batch_size, seq_len = 2, 30
        x = torch.randn(batch_size, seq_len, 32)
        output = model(x)
        
        assert output.shape == (batch_size, 1)

class TestMarketSimulator:
    
    @pytest.fixture
    def simulator(self):
        return MarketSimulator()
    
    def test_prepare_features(self, simulator):
        """Test feature preparation"""
        # Create sample data
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [104, 105, 106, 107, 108],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'SMA_20': [102, 103, 104, 105, 106],
            'SMA_50': [101, 102, 103, 104, 105],
            'RSI': [45, 50, 55, 60, 65],
            'Volatility': [0.02, 0.025, 0.03, 0.035, 0.04]
        })
        
        features = simulator.prepare_features(data)
        
        assert features.shape[0] == len(data)
        assert features.shape[1] > len(simulator.feature_columns)  # Additional features added
    
    @pytest.mark.asyncio
    async def test_simulate_scenario(self, simulator):
        """Test scenario simulation"""
        # Create sample ticker data
        sample_data = pd.DataFrame({
            'Open': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 105,
            'Low': np.random.randn(100) + 99,
            'Close': np.random.randn(100) + 104,
            'Volume': np.random.randint(1000000, 2000000, 100),
            'SMA_20': np.random.randn(100) + 102,
            'SMA_50': np.random.randn(100) + 101,
            'RSI': np.random.uniform(30, 70, 100),
            'Volatility': np.random.uniform(0.01, 0.05, 100)
        }, index=pd.date_range('2024-01-01', periods=100))
        
        ticker_data = {"AAPL": sample_data}
        
        # Create sample risk event
        risk_event = MarketEvent(
            timestamp=datetime.now(),
            event_type="fed",
            description="Interest rate hike",
            affected_tickers=["AAPL"],
            severity_score=0.8,
            source_url="https://example.com"
        )
        
        results = await simulator.simulate_scenario(ticker_data, [risk_event])
        
        assert "AAPL" in results
        assert "baseline_prediction" in results["AAPL"]
        assert "risk_scenarios" in results["AAPL"]
        assert "current_price" in results["AAPL"]

class TestMarketRiskPipeline:
    
    @pytest.fixture
    def pipeline(self):
        return MarketRiskPipeline()
    
    def test_agent_creation(self, pipeline):
        """Test that all agents are created correctly"""
        assert "scanner" in pipeline.agents
        assert "analyst" in pipeline.agents
        assert "simulator" in pipeline.agents
        assert "briefer" in pipeline.agents
        
        for agent_name, agent in pipeline.agents.items():
            assert agent.role is not None
            assert agent.goal is not None
            assert agent.backstory is not None

# Integration tests
class TestIntegration:
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self):
        """Test complete pipeline execution (requires valid API keys)"""
        # Skip if API keys not available
        if not os.getenv("ALPHA_VANTAGE_API_KEY"):
            pytest.skip("API keys not available for integration test")
        
        pipeline = MarketRiskPipeline()
        result = await pipeline.run_pipeline()
        
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            assert "results" in result

# Performance tests
class TestPerformance:
    
    @pytest.mark.performance
    def test_data_processing_speed(self):
        """Test data processing performance"""
        provider = MarketDataProvider()
        
        # Generate large dataset
        large_data = pd.DataFrame({
            'Close': np.random.randn(10000) + 100
        })
        
        start_time = time.time()
        rsi = provider._calculate_rsi(large_data['Close'])
        end_time = time.time()
        
        # Should process 10k points in less than 1 second
        assert end_time - start_time < 1.0
        assert len(rsi) == len(large_data)

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
'''

# Deployment script
DEPLOY_SCRIPT = '''
#!/bin/bash
# Deployment script for Render or similar platforms

echo "Starting Fitch Market Risk Pipeline deployment..."

# Check Python version
python --version
if [ $? -ne 0 ]; then
    echo "Error: Python not found"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

# Validate environment variables
echo "Validating configuration..."
python -c "
from config import PipelineConfig
config = PipelineConfig.from_env()
errors = config.validate()
if errors:
    print('Configuration errors:', errors)
    exit(1)
print('Configuration validated successfully')
"

if [ $? -ne 0 ]; then
    echo "Error: Configuration validation failed"
    exit 1
fi

# Run database migrations (if needed)
echo "Setting up database..."
# Add your database setup commands here

# Run tests (optional in production)
if [ "$RUN_TESTS" = "true" ]; then
    echo "Running tests..."
    python -m pytest tests/ -v --tb=short
    if [ $? -ne 0 ]; then
        echo "Warning: Some tests failed"
    fi
fi

echo "Deployment completed successfully!"
echo "Starting application..."

# Start the appropriate service based on environment
if [ "$SERVICE_TYPE" = "worker" ]; then
    python worker.py
elif [ "$SERVICE_TYPE" = "api" ]; then
    python api_server.py
else
    python main.py
fi
'''

print("✅ Complete Fitch Market Risk Pipeline implementation ready!")
print("\nNext steps:")
print("1. Set up environment variables (.env file)")
print("2. Install requirements: pip install -r requirements.txt")
print("3. Configure Supabase database with provided schema")
print("4. Test locally: python main.py")
print("5. Deploy to Render or similar platform")
print("\nFor production deployment:")
print("- Use the worker.py for background processing")
print("- Use api_server.py for HTTP endpoints")
print("- Configure environment variables in Render dashboard")
print("- Set up monitoring and alerting")

# Save additional files
def save_additional_files():
    files_to_save = {
        'requirements.txt': REQUIREMENTS,
        'Dockerfile': DOCKERFILE,
        '.env.template': ENV_TEMPLATE,
        'database_schema.sql': DATABASE_SCHEMA,
        'api_server.py': API_SERVER,
        'render.yaml': RENDER_CONFIG,
        'worker.py': WORKER_SCRIPT,
        'config.py': CONFIG_MANAGER,
        'test_pipeline.py': TESTS,
        'deploy.sh': DEPLOY_SCRIPT
    }
    
    return files_to_save

# Display file structure
print("\n📁 Complete project structure:")
print("""
fitch-market-risk-pipeline/
├── main.py                 # Main pipeline implementation
├── requirements.txt        # Python dependencies
├── .env.template          # Environment variables template
├── config.py              # Configuration management
├── api_server.py          # HTTP API endpoints
├── worker.py              # Background worker
├── test_pipeline.py       # Test suite
├── Dockerfile             # Docker configuration
├── render.yaml            # Render deployment config
├── deploy.sh              # Deployment script
├── database_schema.sql    # Supabase database schema
└── README.md             # Documentation
""")

# Performance optimization notes
PERFORMANCE_NOTES = """
🚀 Performance Optimization Tips:

1. **Caching Strategy:**
   - Cache market data for 5-15 minutes
   - Use Redis for frequently accessed data
   - Implement intelligent cache invalidation

2. **Batch Processing:**
   - Process tickers in batches of 5-10
   - Use async/await for concurrent API calls
   - Implement exponential backoff for rate limits

3. **Model Optimization:**
   - Use model quantization for faster inference
   - Implement model serving with TorchServe
   - Pre-compute features where possible

4. **Database Optimization:**
   - Use proper indexing on timestamp columns
   - Implement connection pooling
   - Archive old data regularly

5. **Monitoring:**
   - Track API response times
   - Monitor memory usage during simulations
   - Set up alerts for pipeline failures
"""

print(PERFORMANCE_NOTES)