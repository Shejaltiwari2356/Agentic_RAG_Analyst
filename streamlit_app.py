import streamlit as st
import yaml
import os
import json
import re
from dotenv import load_dotenv
from src.agents.financial_auditor import FinancialAuditorAgent
from src.core.parser import PDFParser
from src.utils.cost_tracker import CostTracker

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Financial Auditor AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FinancialAnalystUI:
    def __init__(self):
        self.config_path = "config/config.yaml"
        self.load_config()
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'total_cost_usd' not in st.session_state:
            st.session_state.total_cost_usd = 0.0
        if 'total_cost_inr' not in st.session_state:
            st.session_state.total_cost_inr = 0.0
        if 'ingestion_complete' not in st.session_state:
            st.session_state.ingestion_complete = False
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            st.error(f"Configuration file not found at {self.config_path}")
            self.config = {}
    
    def render_sidebar(self):
        with st.sidebar:
            st.image("https://img.icons8.com/fluency/96/000000/financial-analytics.png", width=80)
            st.title("🔧 Control Panel")
            
            st.divider()
            
            # Document Ingestion Section
            st.subheader("📥 Document Management")
            
            if self.config.get('data', {}).get('pdf_path'):
                pdf_path = self.config['data']['pdf_path']
                if os.path.exists(pdf_path):
                    st.success(f"✅ PDF Found: {os.path.basename(pdf_path)}")
                else:
                    st.warning(f"⚠️ PDF not found at: {pdf_path}")
            
            if st.button("🚀 Ingest Document", type="primary", use_container_width=True):
                self.ingest_document()
            
            if st.session_state.ingestion_complete:
                st.success("✅ Document ingested successfully!")
            
            st.divider()
            
            # Cost Tracking Section
            st.subheader("💰 Cost Tracker")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("USD", f"${st.session_state.total_cost_usd:.4f}")
            with col2:
                st.metric("INR", f"₹{st.session_state.total_cost_inr:.2f}")
            
            st.divider()
            
            # Configuration Display
            with st.expander("⚙️ Configuration"):
                st.json(self.config)
            
            # Clear History Button
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.total_cost_usd = 0.0
                st.session_state.total_cost_inr = 0.0
                st.rerun()
    
    def ingest_document(self):
        with st.spinner("🔄 Processing document... This may take a few minutes."):
            try:
                parser = PDFParser(self.config_path)
                parser.run_smart_ingestion()
                st.session_state.ingestion_complete = True
                st.success("✅ Document ingestion completed successfully!")
            except Exception as e:
                st.error(f"❌ Ingestion failed: {str(e)}")
    
    def extract_chart_data(self, response_text):
        """Extract chart data from response if present"""
        pattern = r'CHART_DATA_START\s*(\{.*?\})\s*CHART_DATA_END'
        match = re.search(pattern, response_text, re.DOTALL)
        
        if match:
            try:
                chart_json = json.loads(match.group(1))
                return chart_json, response_text.replace(match.group(0), "").strip()
            except json.JSONDecodeError:
                return None, response_text
        return None, response_text
    
    def render_chat_message(self, role, content, chart_data=None):
        """Render a chat message with optional chart"""
        with st.chat_message(role):
            st.markdown(content)
            
            if chart_data:
                try:
                    from src.tools.visualizer import VisualizerTool
                    visualizer = VisualizerTool()
                    
                    chart_type = chart_data.get('chart_type', 'bar')
                    labels = chart_data.get('labels', [])
                    values = chart_data.get('values', [])
                    title = chart_data.get('title', 'Financial Data Visualization')
                    
                    visualizer.create_dynamic_chart(chart_type, labels, values, title)
                except Exception as e:
                    st.error(f"Chart rendering error: {str(e)}")
    
    def process_query(self, query):
        """Process user query and get response"""
        try:
            agent = FinancialAuditorAgent(self.config)
            
            with st.spinner("🤔 Analyzing your query..."):
                response = agent.run(query)
                
                # Extract chart data if present
                chart_data, text_response = self.extract_chart_data(response)
                
                # Track costs (if usage data is available)
                # Note: You may need to modify FinancialAuditorAgent to return usage stats
                
                return text_response, chart_data
        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            return None, None
    
    def render_example_queries(self):
        """Render example query buttons"""
        st.subheader("💡 Example Queries")
        
        examples = [
            "What was Apple's net income for 2024 and 2025?",
            "Show me a pie chart of Apple's revenue by product segment",
            "What are the main risk factors mentioned in the 10-K?",
            "Calculate the year-over-year revenue growth",
            "Show me a chart comparing cash flow over the years"
        ]
        
        cols = st.columns(2)
        for idx, example in enumerate(examples):
            with cols[idx % 2]:
                if st.button(example, key=f"example_{idx}", use_container_width=True):
                    st.session_state.selected_example = example
    
    def render_main_content(self):
        # Header
        st.markdown('<div class="main-header">📊 Financial Auditor AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Powered Financial Document Analysis</div>', unsafe_allow_html=True)
        
        # Show example queries if no chat history
        if not st.session_state.chat_history:
            self.render_example_queries()
            st.divider()
        
        # Display chat history
        for message in st.session_state.chat_history:
            self.render_chat_message(
                message['role'], 
                message['content'],
                message.get('chart_data')
            )
        
        # Chat input
        query = st.chat_input("Ask me about financial data...")
        
        # Handle example query selection
        if 'selected_example' in st.session_state:
            query = st.session_state.selected_example
            del st.session_state.selected_example
        
        if query:
            # Add user message to chat
            st.session_state.chat_history.append({
                'role': 'user',
                'content': query
            })
            
            # Display user message
            self.render_chat_message('user', query)
            
            # Get AI response
            response_text, chart_data = self.process_query(query)
            
            if response_text:
                # Add assistant message to chat
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response_text,
                    'chart_data': chart_data
                })
                
                # Display assistant message
                self.render_chat_message('assistant', response_text, chart_data)
                
                st.rerun()
    
    def run(self):
        """Main application entry point"""
        self.render_sidebar()
        self.render_main_content()

# Run the application
if __name__ == "__main__":
    app = FinancialAnalystUI()
    app.run()