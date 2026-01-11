"""
Streamlit Dashboard for Federated Multi-Modal Recommendations
Interactive UI with explainability features
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Federated Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

# ============================================================================
# Utility Functions
# ============================================================================

def call_api(endpoint: str, method: str = "GET", data: dict = None):
    """Call API endpoint"""
    try:
        url = f"{API_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, params=data)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure server is running at http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None


def plot_fusion_weights(weights: dict):
    """Plot fusion weights as pie chart"""
    
    labels = ['Text', 'Image', 'Behavior']
    values = [weights['text'], weights['image'], weights['behavior']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="Modality Fusion Weights",
        showlegend=True,
        height=350,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig


def plot_contribution_bars(items: list):
    """Plot modality contributions for top items"""
    
    df = pd.DataFrame([
        {
            'Item': item['name'][:20] + '...' if len(item['name']) > 20 else item['name'],
            'Text': item['text_contribution'],
            'Image': item['image_contribution'],
            'Behavior': item['behavior_contribution']
        }
        for item in items[:5]
    ])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Text',
        x=df['Item'],
        y=df['Text'],
        marker_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Bar(
        name='Image',
        x=df['Item'],
        y=df['Image'],
        marker_color='#4ECDC4'
    ))
    
    fig.add_trace(go.Bar(
        name='Behavior',
        x=df['Item'],
        y=df['Behavior'],
        marker_color='#45B7D1'
    ))
    
    fig.update_layout(
        title="Modality Contributions per Item",
        barmode='stack',
        xaxis_title="Items",
        yaxis_title="Contribution Weight",
        height=400,
        showlegend=True
    )
    
    return fig


def plot_recommendation_scores(items: list):
    """Plot recommendation scores"""
    
    names = [item['name'][:30] + '...' if len(item['name']) > 30 else item['name'] 
             for item in items[:10]]
    scores = [item['score'] for item in items[:10]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=names,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score")
            )
        )
    ])
    
    fig.update_layout(
        title="Top Recommendations by Score",
        xaxis_title="Recommendation Score",
        yaxis_title="Items",
        height=500,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    
    # Header
    st.title("🎯 Federated Multi-Modal Recommendation System")
    st.markdown("**Personalized recommendations with privacy-preserving federated learning**")
    
    # Check API health
    health = call_api("/health")
    
    if not health:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API status
        st.subheader("🔌 API Status")
        if health['status'] == 'healthy':
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
        
        st.metric("Total Items", health['num_items'])
        st.metric("Total Users", health['num_users'])
        
        st.divider()
        
        # User selection
        st.subheader("👤 User Selection")
        user_id = st.number_input(
            "User ID",
            min_value=0,
            max_value=health['num_users'] - 1,
            value=0,
            step=1
        )
        
        # Recommendation settings
        st.subheader("🎚️ Recommendation Settings")
        top_k = st.slider("Number of recommendations", 5, 20, 10)
        show_explainability = st.checkbox("Show Explainability", value=True)
        
        st.divider()
        
        # Get recommendations button
        if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
            st.session_state['refresh'] = True
    
    # Main content
    tabs = st.tabs(["📊 Recommendations", "👤 User Profile", "📈 Statistics"])
    
    # ========================================================================
    # Tab 1: Recommendations
    # ========================================================================
    with tabs[0]:
        st.header("Personalized Recommendations")
        
        # Get recommendations
        with st.spinner("Generating recommendations..."):
            rec_response = call_api(
                "/recommend",
                method="POST",
                data={
                    "user_id": user_id,
                    "top_k": top_k,
                    "explain": show_explainability
                }
            )
        
        if rec_response:
            
            # User preference type
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.metric("User Preference Type", rec_response['user_preference_type'])
            
            with col2:
                st.metric("Processing Time", f"{rec_response['processing_time_ms']:.2f} ms")
            
            with col3:
                st.metric("Items Returned", len(rec_response['recommendations']))
            
            st.divider()
            
            # Explainability section
            if show_explainability:
                st.subheader("🔍 Explainability")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Fusion weights pie chart
                    fig_pie = plot_fusion_weights(rec_response['fusion_weights'])
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Recommendation scores
                    fig_scores = plot_recommendation_scores(rec_response['recommendations'])
                    st.plotly_chart(fig_scores, use_container_width=True)
                
                # Modality contributions
                st.subheader("Modality Contributions per Item")
                fig_contrib = plot_contribution_bars(rec_response['recommendations'])
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                st.divider()
            
            # Recommendations table
            st.subheader("🎯 Recommended Items")
            
            # Convert to DataFrame
            items_data = []
            for item in rec_response['recommendations']:
                items_data.append({
                    'Rank': item['rank'],
                    'Item ID': item['item_id'],
                    'Name': item['name'],
                    'Category': item['category'],
                    'Score': f"{item['score']:.4f}",
                    'Rating': f"⭐ {item['avg_rating']:.1f} ({item['num_ratings']} reviews)",
                    'Price': f"${item['price']:.2f}",
                    'Brand': item['brand']
                })
            
            df_items = pd.DataFrame(items_data)
            
            # Display with styling
            st.dataframe(
                df_items,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Rank': st.column_config.NumberColumn(format="%d"),
                    'Score': st.column_config.NumberColumn(format="%.4f")
                }
            )
            
            # Detailed view
            with st.expander("📋 Detailed Item Information"):
                selected_item_idx = st.selectbox(
                    "Select item to view details",
                    range(len(rec_response['recommendations'])),
                    format_func=lambda i: f"#{i+1}: {rec_response['recommendations'][i]['name']}"
                )
                
                selected_item = rec_response['recommendations'][selected_item_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Item Information**")
                    st.write(f"- **Item ID**: {selected_item['item_id']}")
                    st.write(f"- **Name**: {selected_item['name']}")
                    st.write(f"- **Category**: {selected_item['category']}")
                    st.write(f"- **Brand**: {selected_item['brand']}")
                    st.write(f"- **Price**: ${selected_item['price']:.2f}")
                
                with col2:
                    st.write("**Recommendation Details**")
                    st.write(f"- **Rank**: #{selected_item['rank']}")
                    st.write(f"- **Score**: {selected_item['score']:.4f}")
                    st.write(f"- **Rating**: ⭐ {selected_item['avg_rating']:.1f}")
                    st.write(f"- **Reviews**: {selected_item['num_ratings']}")
                
                if show_explainability:
                    st.write("**Modality Contributions**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Text", f"{selected_item['text_contribution']:.4f}")
                    with col2:
                        st.metric("Image", f"{selected_item['image_contribution']:.4f}")
                    with col3:
                        st.metric("Behavior", f"{selected_item['behavior_contribution']:.4f}")
    
    # ========================================================================
    # Tab 2: User Profile
    # ========================================================================
    with tabs[1]:
        st.header("User Profile")
        
        with st.spinner("Loading user profile..."):
            user_profile = call_api(f"/user/{user_id}")
        
        if user_profile:
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📋 Basic Information")
                st.write(f"**User ID**: {user_profile['user_id']}")
                st.write(f"**Age**: {user_profile['age']}")
                st.write(f"**Registration Date**: {user_profile['registration_date']}")
                
                st.divider()
                
                st.subheader("🎯 Preferences")
                st.write(f"**Preference Type**: {user_profile['preference_type']}")
                st.write(f"**Preferred Categories**:")
                for cat in user_profile['preferred_categories']:
                    st.write(f"  - {cat}")
            
            with col2:
                st.subheader("🔮 Learned Modality Preferences")
                
                # Fusion weights
                fig_user_weights = plot_fusion_weights(user_profile['fusion_weights'])
                st.plotly_chart(fig_user_weights, use_container_width=True)
                
                st.divider()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Text Weight",
                        f"{user_profile['fusion_weights']['text']:.4f}",
                        help="How much user prefers text-based features"
                    )
                
                with col2:
                    st.metric(
                        "Image Weight",
                        f"{user_profile['fusion_weights']['image']:.4f}",
                        help="How much user prefers image-based features"
                    )
                
                with col3:
                    st.metric(
                        "Behavior Weight",
                        f"{user_profile['fusion_weights']['behavior']:.4f}",
                        help="How much user prefers popularity/behavior features"
                    )
    
    # ========================================================================
    # Tab 3: Statistics
    # ========================================================================
    with tabs[2]:
        st.header("System Statistics")
        
        with st.spinner("Loading statistics..."):
            stats = call_api("/stats")
        
        if stats:
            
            # Overall stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Items", stats['total_items'])
            
            with col2:
                st.metric("Total Users", stats['total_users'])
            
            with col3:
                st.metric("Avg Rating", f"{stats['avg_item_rating']:.2f}")
            
            with col4:
                st.metric("Total Ratings", f"{stats['total_ratings']:,}")
            
            st.divider()
            
            # Category distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📦 Item Categories")
                
                categories = pd.DataFrame(
                    list(stats['categories'].items()),
                    columns=['Category', 'Count']
                ).sort_values('Count', ascending=False)
                
                fig_cat = px.bar(
                    categories,
                    x='Category',
                    y='Count',
                    title="Items per Category",
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                st.subheader("👥 User Preference Types")
                
                prefs = pd.DataFrame(
                    list(stats['preference_types'].items()),
                    columns=['Preference Type', 'Count']
                )
                
                fig_pref = px.pie(
                    prefs,
                    names='Preference Type',
                    values='Count',
                    title="User Preference Distribution",
                    hole=0.3
                )
                st.plotly_chart(fig_pref, use_container_width=True)
            
            st.divider()
            
            # Milvus stats
            if 'milvus' in stats:
                st.subheader("🗄️ Vector Database Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Collection Name", stats['milvus']['collection_name'])
                
                with col2:
                    st.metric("Stored Embeddings", stats['milvus']['num_entities'])


# ============================================================================
# Run Dashboard
# ============================================================================

if __name__ == "__main__":
    main()