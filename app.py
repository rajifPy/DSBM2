import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import flask
import os
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Inisialisasi app
app = dash.Dash(__name__, server=flask.Flask(__name__))
app.title = "Dashboard Visualisasi Volume Bongkar Muat antar Pulau di Pelabuhan Utama Indonesia - BPS"

# Layout aplikasi
app.layout = html.Div([
    # Header Section
    html.Div([
        html.Div([
            html.H1("📊 Dashboard Visualisasi Volume Bongkar Muat 📊", className="header-title"),
            html.H2("Pelabuhan Utama Indonesia", className="header-subtitle"),
            html.P("Analisis Time Series & Exploratory Data Visualization", 
                   className="header-description"),
            html.Div([
                html.Span("", className="header-icon"),
                html.Span("Interactive Analytics Dashboard", className="header-tag"),
            ], className="header-meta")
        ], className="header-content"),
    ], className="header"),
    
    # Control Panel
    html.Div([
        html.H3("🎛️ Panel Kontrol", className="section-title"),
        html.Div([
            # Filters Row 1
            html.Div([
                html.Div([
                    html.Label("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Pilih Pelabuhan", className="control-label"),
                    dcc.Dropdown(
                        id='port-selector',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="Pilih satu atau lebih pelabuhan...",
                        className="modern-dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("📅 Pilih Tahun", className="control-label"),
                    dcc.Dropdown(
                        id='year-selector',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="Pilih tahun analisis...",
                        className="modern-dropdown"
                    )
                ], className="control-item"),
            ], className="controls-row"),
            
            # Filters Row 2
            html.Div([
                html.Div([
                    html.Label("📆 Rentang Tanggal", className="control-label"),
                    dcc.DatePickerRange(
                        id='date-range',
                        min_date_allowed=datetime(2006, 1, 1),
                        max_date_allowed=datetime(2024, 12, 31),
                        start_date=datetime(2020, 1, 1),
                        end_date=datetime(2024, 12, 31),
                        display_format='DD/MM/YYYY',
                        className="modern-date-picker"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("📊 Jenis Analisis", className="control-label"),
                    dcc.RadioItems(
                        id='analysis-type',
                        options=[
                            {'label': '📈 Time Series', 'value': 'timeseries'},
                            {'label': '📉 Trend Analysis', 'value': 'trend'},
                            {'label': '🔍 Statistical', 'value': 'statistical'}
                        ],
                        value='timeseries',
                        className="modern-radio",
                        inline=True
                    )
                ], className="control-item"),
            ], className="controls-row"),
            
            # Action Buttons
            html.Div([
                html.Button('🔄 Muat Data', id='load-data-button', n_clicks=0, className='primary-button'),
                html.Button('📊 Refresh Charts', id='refresh-button', n_clicks=0, className='secondary-button'),
                html.Button('📋 Export Data', id='export-button', n_clicks=0, className='tertiary-button'),
            ], className="button-group"),
            
            dcc.Loading(
                id="loading-data",
                type="dot",
                color="#1a5fb4",
                children=html.Div(id="loading-output", className="loading-container")
            )
        ], className="control-panel"),
    ], className="controls-section"),
    
    # Quick Stats Overview
    html.Div([
        html.H3("📈 Statistik Ringkas", className="section-title"),
        html.Div(id='quick-stats', className="quick-stats-container")
    ], className="stats-overview"),
    
    # Main Charts Section
    html.Div([
        html.H3("📊 Visualisasi Utama", className="section-title"),
        
        # Primary Charts Row
        html.Div([
            html.Div([
                dcc.Graph(id='time-series-chart', className="main-chart")
            ], className="chart-container primary-chart"),
            
            html.Div([
                dcc.Graph(id='volume-pie-chart', className="main-chart")
            ], className="chart-container secondary-chart"),
        ], className="charts-row primary-row"),
        
        # Secondary Charts Row
        html.Div([
            html.Div([
                dcc.Graph(id='comparison-chart', className="main-chart")
            ], className="chart-container full-width"),
        ], className="charts-row secondary-row"),
        
        # Advanced Analytics Row
        html.Div([
            html.Div([
                dcc.Graph(id='correlation-heatmap', className="main-chart")
            ], className="chart-container"),
            
            html.Div([
                dcc.Graph(id='distribution-chart', className="main-chart")
            ], className="chart-container"),
        ], className="charts-row tertiary-row"),
        
        # Trend Analysis Row
        html.Div([
            html.Div([
                dcc.Graph(id='seasonal-decomposition', className="main-chart")
            ], className="chart-container full-width"),
        ], className="charts-row quaternary-row"),
    ], className="charts-section"),
    
    # Detailed Statistics Section
    html.Div([
        html.H3("📋 Analisis Statistik Detail", className="section-title"),
        html.Div([
            html.Div([
                html.H4("🎯 Statistik Deskriptif"),
                html.Div(id='detailed-stats', className="stats-grid")
            ], className="stats-card"),
            
            html.Div([
                html.H4("📊 Tabel Data"),
                html.Div(id='data-table-container')
            ], className="table-card"),
        ], className="detailed-analysis-row")
    ], className="detailed-stats-section"),
    
    # Footer Information
    html.Div([
        html.Div([
            html.Div([
                html.H4("ℹ️ Informasi Dataset"),
                html.P("Sumber: Badan Pusat Statistik (BPS) Indonesia"),
                html.P("Dataset: Volume Muat Barang Angkutan Laut di Pelabuhan Utama"),
                html.P("Periode: 2006 - 2024"),
                html.P("Unit: Ton"),
            ], className="info-card"),
            
            html.Div([
                html.H4("🔧 Fitur Dashboard"),
                html.Ul([
                    html.Li("Time Series Analysis dengan tren dan pola musiman"),
                    html.Li("Analisis korelasi antar pelabuhan"),
                    html.Li("Statistik deskriptif"),
                    html.Li("Visualisasi interaktif dan responsive"),
                    html.Li("Export data dan filtering lanjutan"),
                ], className="feature-list")
            ], className="info-card"),
        ], className="footer-content")
    ], className="footer-section"),
    
    # Data Storage
    dcc.Store(id='data-store'),
    dcc.Store(id='processed-data-store'),
], className="main-container")

# Fungsi untuk memuat data dari CSV
def load_csv_data():
    try:
        # Baca file CSV
        df = pd.read_csv('data/data_bongkar_muat_4_pelabuhan utama.csv')
        
        # Ubah format tanggal
        df['tgl'] = pd.to_datetime(df['tgl'])
        
        # Ubah format data dari wide ke long
        df_long = df.melt(id_vars=['tgl'], var_name='pelabuhan', value_name='volume')
        
        # Ekstrak tahun, bulan, dan kuartal
        df_long['tahun'] = df_long['tgl'].dt.year
        df_long['bulan'] = df_long['tgl'].dt.month
        df_long['bulan_nama'] = df_long['tgl'].dt.month_name()
        df_long['kuartal'] = df_long['tgl'].dt.quarter
        df_long['hari_dalam_minggu'] = df_long['tgl'].dt.day_name()
        
        return df_long
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return get_enhanced_sample_data()

def get_enhanced_sample_data():
    # Buat data sampel yang lebih realistis dengan pola seasonal
    np.random.seed(42)
    
    sample_dates = pd.date_range(start='2006-01-01', end='2024-12-01', freq='MS')
    sample_data = []
    
    pelabuhan_list = ['Belawan', 'Tanjung Priok', 'Tanjung Perak', 'Makassar']
    base_volumes = {'Belawan': 800000, 'Tanjung Priok': 1200000, 'Tanjung Perak': 900000, 'Makassar': 600000}
    
    for i, date in enumerate(sample_dates):
        for pelabuhan in pelabuhan_list:
            # Tambahkan tren naik, pola seasonal, dan noise
            trend = i * 1000
            seasonal = 100000 * np.sin(2 * np.pi * date.month / 12)
            noise = np.random.normal(0, 50000)
            
            volume = max(0, base_volumes[pelabuhan] + trend + seasonal + noise)
            
            sample_data.append({
                'tgl': date,
                'pelabuhan': pelabuhan,
                'volume': volume,
                'tahun': date.year,
                'bulan': date.month,
                'bulan_nama': date.strftime('%B'),
                'kuartal': (date.month - 1) // 3 + 1,
                'hari_dalam_minggu': date.strftime('%A')
            })
    
    return pd.DataFrame(sample_data)

# Callback untuk fetch data
@app.callback(
    [Output('data-store', 'data'),
     Output('port-selector', 'options'),
     Output('port-selector', 'value'),
     Output('year-selector', 'options'),
     Output('year-selector', 'value'),
     Output('loading-output', 'children')],
    [Input('load-data-button', 'n_clicks')],
    prevent_initial_call=True
)
def fetch_data(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return [], [], [], [], [], ""
    
    try:
        # Mengambil data dari CSV
        data = load_csv_data()
        
        if data is None or data.empty:
            return [], [], [], [], [], "❌ Gagal memuat data"
        
        # Ambil daftar pelabuhan
        ports = [{'label': f"🏴󠁧󠁢󠁥󠁮󠁧󠁿 {port}", 'value': port} for port in sorted(data['pelabuhan'].unique())]
        default_ports = list(data['pelabuhan'].unique())
        
        # Ambil daftar tahun
        years = [{'label': f"📅 {year}", 'value': year} for year in sorted(data['tahun'].unique())]
        recent_years = sorted(data['tahun'].unique())[-3:] if len(data['tahun'].unique()) >= 3 else list(data['tahun'].unique())
        
        success_msg = html.Div([
            html.Span("✅ Data berhasil dimuat!", className="success-msg"),
            html.Br(),
            html.Small(f"📊 {len(data):,} records | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 {len(data['pelabuhan'].unique())} pelabuhan")
        ])
        
        return data.to_dict('records'), ports, default_ports, years, recent_years, success_msg
    except Exception as e:
        error_msg = html.Div([
            html.Span(f"❌ Error: {str(e)}", className="error-msg"),
        ])
        return [], [], [], [], [], error_msg

# Callback untuk quick stats
@app.callback(
    Output('quick-stats', 'children'),
    [Input('data-store', 'data'),
     Input('port-selector', 'value'),
     Input('year-selector', 'value')],
    prevent_initial_call=True
)
def update_quick_stats(data, selected_ports, selected_years):
    if not data:
        return [html.Div("📊 Muat data untuk melihat statistik", className="placeholder-text")]
    
    try:
        df = pd.DataFrame(data)
        df['tgl'] = pd.to_datetime(df['tgl'])
        
        # Filter data
        filtered_df = df.copy()
        if selected_ports:
            filtered_df = filtered_df[filtered_df['pelabuhan'].isin(selected_ports)]
        if selected_years:
            filtered_df = filtered_df[filtered_df['tahun'].isin(selected_years)]
        
        if filtered_df.empty:
            return [html.Div("⚠️ Tidak ada data dengan filter yang dipilih", className="warning-text")]
        
        # Hitung statistik
        total_volume = filtered_df['volume'].sum()
        avg_volume = filtered_df['volume'].mean()
        growth_rate = 0
        
        # Hitung pertumbuhan year-over-year jika ada data multi tahun
        if len(filtered_df['tahun'].unique()) > 1:
            yearly_totals = filtered_df.groupby('tahun')['volume'].sum()
            if len(yearly_totals) >= 2:
                growth_rate = ((yearly_totals.iloc[-1] - yearly_totals.iloc[-2]) / yearly_totals.iloc[-2]) * 100
        
        max_month = filtered_df.loc[filtered_df['volume'].idxmax()]
        
        return [
            html.Div([
                html.Div("📦", className="stat-icon"),
                html.Div([
                    html.H3(f"{total_volume/1_000_000:.1f}M"),
                    html.P("Total Volume (Ton)")
                ], className="stat-content")
            ], className="quick-stat-card primary"),
            
            html.Div([
                html.Div("📊", className="stat-icon"),
                html.Div([
                    html.H3(f"{avg_volume/1_000:.0f}K"),
                    html.P("Rata-rata Volume")
                ], className="stat-content")
            ], className="quick-stat-card secondary"),
            
            html.Div([
                html.Div("📈" if growth_rate >= 0 else "📉", className="stat-icon"),
                html.Div([
                    html.H3(f"{growth_rate:+.1f}%"),
                    html.P("Pertumbuhan YoY")
                ], className="stat-content")
            ], className="quick-stat-card tertiary"),
            
            html.Div([
                html.Div("🏆", className="stat-icon"),
                html.Div([
                    html.H3(f"{max_month['pelabuhan'][:8]}"),
                    html.P("Peringkat Pelabuhan")
                ], className="stat-content")
            ], className="quick-stat-card quaternary"),
        ]
    except Exception as e:
        return [html.Div(f"❌ Error: {str(e)}", className="error-text")]

# Callback untuk time series chart dengan analisis yang lebih canggih
@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('data-store', 'data'),
     Input('port-selector', 'value'),
     Input('year-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_time_series(data, selected_ports, selected_years, start_date, end_date, analysis_type):
    if not data:
        return go.Figure().add_annotation(text="📊 Muat data untuk melihat grafik", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    try:
        df = pd.DataFrame(data)
        df['tgl'] = pd.to_datetime(df['tgl'])
        
        # Filter data
        filtered_df = df.copy()
        if selected_ports:
            filtered_df = filtered_df[filtered_df['pelabuhan'].isin(selected_ports)]
        if selected_years:
            filtered_df = filtered_df[filtered_df['tahun'].isin(selected_years)]
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['tgl'] >= start_date) & (filtered_df['tgl'] <= end_date)]
        
        if filtered_df.empty:
            return go.Figure().add_annotation(text="⚠️ Tidak ada data dengan filter yang dipilih", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Buat grafik berdasarkan jenis analisis
        if analysis_type == 'timeseries':
            fig = create_enhanced_timeseries(filtered_df)
        elif analysis_type == 'trend':
            fig = create_trend_analysis(filtered_df)
        else:
            fig = create_statistical_chart(filtered_df)
        
        fig.update_layout(
            title="📈 Analisis Time Series Volume Bongkar Muat",
            template="plotly_white",
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"❌ Error: {str(e)}", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

def create_enhanced_timeseries(df):
    """Buat time series chart dengan moving average dan trend line"""
    time_data = df.groupby(['tgl', 'pelabuhan'])['volume'].sum().reset_index()
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, pelabuhan in enumerate(time_data['pelabuhan'].unique()):
        port_data = time_data[time_data['pelabuhan'] == pelabuhan].sort_values('tgl')
        
        # Line utama
        fig.add_trace(go.Scatter(
            x=port_data['tgl'],
            y=port_data['volume'],
            mode='lines+markers',
            name=f"🏴󠁧󠁢󠁥󠁮󠁧󠁿 {pelabuhan}",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
            hovertemplate="<b>%{fullData.name}</b><br>Tanggal: %{x}<br>Volume: %{y:,.0f} ton<extra></extra>"
        ))
        
        # Moving average 6 bulan jika data cukup
        if len(port_data) >= 6:
            port_data['ma_6'] = port_data['volume'].rolling(window=6, center=True).mean()
            fig.add_trace(go.Scatter(
                x=port_data['tgl'],
                y=port_data['ma_6'],
                mode='lines',
                name=f"📊 MA-6 {pelabuhan}",
                line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
                opacity=0.7,
                hovertemplate="<b>Moving Average</b><br>%{y:,.0f} ton<extra></extra>"
            ))
    
    return fig

def create_trend_analysis(df):
    """Buat analisis tren dengan regression line"""
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('📈 Trend Analysis', '📊 Growth Rate'),
                       vertical_spacing=0.15)
    
    colors = px.colors.qualitative.Set1
    
    for i, pelabuhan in enumerate(df['pelabuhan'].unique()):
        port_data = df[df['pelabuhan'] == pelabuhan].groupby('tgl')['volume'].sum().reset_index()
        port_data = port_data.sort_values('tgl')
        
        # Data asli
        fig.add_trace(go.Scatter(
            x=port_data['tgl'], y=port_data['volume'],
            mode='lines', name=pelabuhan,
            line=dict(color=colors[i % len(colors)]),
        ), row=1, col=1)
        
        # Trend line menggunakan regresi linear
        x_numeric = pd.to_numeric(port_data['tgl'])
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, port_data['volume'])
        trend_line = slope * x_numeric + intercept
        
        fig.add_trace(go.Scatter(
            x=port_data['tgl'], y=trend_line,
            mode='lines', name=f"Trend {pelabuhan}",
            line=dict(color=colors[i % len(colors)], dash='dash'),
            showlegend=False
        ), row=1, col=1)
        
        # Growth rate
        port_data['growth_rate'] = port_data['volume'].pct_change() * 100
        fig.add_trace(go.Scatter(
            x=port_data['tgl'], y=port_data['growth_rate'],
            mode='lines', name=f"Growth {pelabuhan}",
            line=dict(color=colors[i % len(colors)]),
            showlegend=False
        ), row=2, col=1)
    
    fig.update_yaxes(title_text="Volume (Ton)", row=1, col=1)
    fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="Tanggal", row=2, col=1)
    
    return fig

def create_statistical_chart(df):
    """Buat box plot dan violin plot untuk analisis statistik"""
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('📊 Box Plot Distribution', '🎻 Violin Plot Distribution'))
    
    colors = px.colors.qualitative.Set1
    
    for i, pelabuhan in enumerate(df['pelabuhan'].unique()):
        port_data = df[df['pelabuhan'] == pelabuhan]['volume']
        
        # Box plot
        fig.add_trace(go.Box(
            y=port_data, name=pelabuhan,
            marker_color=colors[i % len(colors)],
            showlegend=False
        ), row=1, col=1)
        
        # Violin plot
        fig.add_trace(go.Violin(
            y=port_data, name=pelabuhan,
            fillcolor=colors[i % len(colors)],
            line_color=colors[i % len(colors)],
            showlegend=False
        ), row=1, col=2)
    
    fig.update_yaxes(title_text="Volume (Ton)")
    return fig

# Callback untuk pie chart yang lebih interaktif
@app.callback(
    Output('volume-pie-chart', 'figure'),
    [Input('data-store', 'data'),
     Input('port-selector', 'value'),
     Input('year-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')],
    prevent_initial_call=True
)
def update_enhanced_pie_chart(data, selected_ports, selected_years, start_date, end_date):
    if not data:
        return go.Figure().add_annotation(text="📊 Muat data untuk melihat distribusi", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    try:
        df = pd.DataFrame(data)
        df['tgl'] = pd.to_datetime(df['tgl'])
        
        # Filter data
        filtered_df = df.copy()
        if selected_ports:
            filtered_df = filtered_df[filtered_df['pelabuhan'].isin(selected_ports)]
        if selected_years:
            filtered_df = filtered_df[filtered_df['tahun'].isin(selected_years)]
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['tgl'] >= start_date) & (filtered_df['tgl'] <= end_date)]
        
        if filtered_df.empty:
            return go.Figure().add_annotation(text="⚠️ Tidak ada data dengan filter yang dipilih", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Aggregate by port
        port_totals = filtered_df.groupby('pelabuhan')['volume'].sum().reset_index()
        port_totals['percentage'] = (port_totals['volume'] / port_totals['volume'].sum()) * 100
        
        fig = go.Figure(data=[go.Pie(
            labels=[f"{port}" for port in port_totals['pelabuhan']],
            values=port_totals['volume'],
            hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f} ton<br>Persentase: %{percent}<extra></extra>",
            textinfo='label+percent',
            textposition='auto',
            marker=dict(
                colors=px.colors.qualitative.Set1,
                line=dict(color='white', width=2)
            ),
            pull=[0.05] * len(port_totals)  # Sedikit tarik keluar untuk efek visual
        )])
        
        fig.update_layout(
            title="🥧 Distribusi Volume per Pelabuhan",
            showlegend=True,
            height=500,
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"❌ Error: {str(e)}", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# Callback untuk comparison chart yang lebih canggih
@app.callback(
    Output('comparison-chart', 'figure'),
    [Input('data-store', 'data'),
     Input('port-selector', 'value'),
     Input('year-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')],
    prevent_initial_call=True
)
def update_enhanced_comparison_chart(data, selected_ports, selected_years, start_date, end_date):
    if not data:
        return go.Figure().add_annotation(text="📊 Muat data untuk melihat perbandingan", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    try:
        df = pd.DataFrame(data)
        df['tgl'] = pd.to_datetime(df['tgl'])
        
        # Filter data
        filtered_df = df.copy()
        if selected_ports:
            filtered_df = filtered_df[filtered_df['pelabuhan'].isin(selected_ports)]
        if selected_years:
            filtered_df = filtered_df[filtered_df['tahun'].isin(selected_years)]
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['tgl'] >= start_date) & (filtered_df['tgl'] <= end_date)]
        
        if filtered_df.empty:
            return go.Figure().add_annotation(text="⚠️ Tidak ada data dengan filter yang dipilih", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Perbandingan berdasarkan tahun dan kuartal
        comparison_data = filtered_df.groupby(['tahun', 'kuartal', 'pelabuhan'])['volume'].sum().reset_index()
        comparison_data['period'] = comparison_data['tahun'].astype(str) + '-Q' + comparison_data['kuartal'].astype(str)
        
        fig = px.bar(comparison_data, x='period', y='volume', color='pelabuhan',
                     title='📊 Perbandingan Volume per Kuartal',
                     barmode='group',
                     hover_data={'volume': ':,.0f'},
                     color_discrete_sequence=px.colors.qualitative.Set1)
        
        fig.update_layout(
            xaxis_title="📅 Period",
            yaxis_title="📦 Volume (Ton)",
            template="plotly_white",
            height=500,
            showlegend=True
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"❌ Error: {str(e)}", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# Callback untuk correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('data-store', 'data'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')],
    prevent_initial_call=True
)
def update_correlation_heatmap(data, start_date, end_date):
    if not data:
        return go.Figure().add_annotation(text="📊 Muat data untuk melihat korelasi", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    try:
        df = pd.DataFrame(data)
        df['tgl'] = pd.to_datetime(df['tgl'])
        
        # Filter berdasarkan tanggal
        if start_date and end_date:
            df = df[(df['tgl'] >= start_date) & (df['tgl'] <= end_date)]
        
        # Pivot data untuk korelasi
        pivot_df = df.pivot_table(index='tgl', columns='pelabuhan', values='volume', aggfunc='sum')
        
        if pivot_df.empty:
            return go.Figure().add_annotation(text="⚠️ Tidak ada data untuk analisis korelasi", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Hitung korelasi
        corr_matrix = pivot_df.corr()
        
        # Buat heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[f"🏴󠁧󠁢󠁥󠁮󠁧󠁿 {col}" for col in corr_matrix.columns],
            y=[f"🏴󠁧󠁢󠁥󠁮󠁧󠁿 {idx}" for idx in corr_matrix.index],
            colorscale='RdYlBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hovetemplate="<b>%{x} vs %{y}</b><br>Korelasi: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="🔗 Matriks Korelasi antar Pelabuhan",
            template="plotly_white",
            height=400
        )
        
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"❌ Error: {str(e)}", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# Callback untuk distribution chart
@app.callback(
    Output('distribution-chart', 'figure'),
    [Input('data-store', 'data'),
     Input('port-selector', 'value'),
     Input('year-selector', 'value')],
    prevent_initial_call=True
)
def update_distribution_chart(data, selected_ports, selected_years):
    if not data:
        return go.Figure().add_annotation(text="📊 Muat data untuk melihat distribusi", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    try:
        df = pd.DataFrame(data)
        
        # Filter data
        if selected_ports:
            df = df[df['pelabuhan'].isin(selected_ports)]
        if selected_years:
            df = df[df['tahun'].isin(selected_years)]
        
        if df.empty:
            return go.Figure().add_annotation(text="⚠️ Tidak ada data dengan filter yang dipilih", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Buat histogram dengan overlay
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, pelabuhan in enumerate(df['pelabuhan'].unique()):
            port_data = df[df['pelabuhan'] == pelabuhan]['volume']
            
            fig.add_trace(go.Histogram(
                x=port_data,
                name=f"{pelabuhan}",
                opacity=0.7,
                marker_color=colors[i % len(colors)],
                nbinsx=30
            ))
        
        fig.update_layout(
            title="📈 Distribusi Frekuensi Volume",
            xaxis_title="📦 Volume (Ton)",
            yaxis_title="📊 Frekuensi",
            barmode='overlay',
            template="plotly_white",
            height=400
        )
        
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"❌ Error: {str(e)}", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# Callback untuk seasonal decomposition
@app.callback(
    Output('seasonal-decomposition', 'figure'),
    [Input('data-store', 'data'),
     Input('port-selector', 'value')],
    prevent_initial_call=True
)
def update_seasonal_decomposition(data, selected_ports):
    if not data:
        return go.Figure().add_annotation(text="📊 Muat data untuk analisis seasonal", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    try:
        df = pd.DataFrame(data)
        df['tgl'] = pd.to_datetime(df['tgl'])
        
        # Filter port (ambil yang pertama jika multiple)
        if selected_ports:
            df = df[df['pelabuhan'] == selected_ports[0]]
        else:
            df = df[df['pelabuhan'] == df['pelabuhan'].unique()[0]]
        
        # Agregasi bulanan
        monthly_data = df.groupby([df['tgl'].dt.to_period('M')])['volume'].sum().reset_index()
        monthly_data['tgl'] = monthly_data['tgl'].dt.to_timestamp()
        
        if len(monthly_data) < 24:  # Butuh minimal 2 tahun untuk seasonal analysis
            return go.Figure().add_annotation(text="⚠️ Data tidak cukup untuk analisis seasonal (minimal 24 bulan)", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Simple seasonal analysis - perbandingan rata-rata bulanan
        monthly_data['bulan'] = monthly_data['tgl'].dt.month
        seasonal_pattern = monthly_data.groupby('bulan')['volume'].mean()
        
        # Buat subplot untuk seasonal pattern
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('📈 Data Time Series', '🔄 Pola Seasonal (Rata-rata Bulanan)'),
                           vertical_spacing=0.15)
        
        # Time series asli
        fig.add_trace(go.Scatter(
            x=monthly_data['tgl'],
            y=monthly_data['volume'],
            mode='lines+markers',
            name='Volume Bulanan',
            line=dict(color='#1f77b4', width=2)
        ), row=1, col=1)
        
        # Pola seasonal
        bulan_nama = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig.add_trace(go.Bar(
            x=bulan_nama,
            y=seasonal_pattern.values,
            name='Rata-rata Bulanan',
            marker_color='#ff7f0e'
        ), row=2, col=1)
        
        fig.update_xaxes(title_text="📅 Tanggal", row=1, col=1)
        fig.update_xaxes(title_text="📅 Bulan", row=2, col=1)
        fig.update_yaxes(title_text="📦 Volume (Ton)", row=1, col=1)
        fig.update_yaxes(title_text="📦 Rata-rata Volume", row=2, col=1)
        
        fig.update_layout(
            title=f"🔄 Analisis Seasonal - {selected_ports[0] if selected_ports else 'All Ports'}",
            template="plotly_white",
            height=600,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"❌ Error: {str(e)}", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# Callback untuk detailed stats
@app.callback(
    Output('detailed-stats', 'children'),
    [Input('data-store', 'data'),
     Input('port-selector', 'value'),
     Input('year-selector', 'value')],
    prevent_initial_call=True
)
def update_detailed_stats(data, selected_ports, selected_years):
    if not data:
        return [html.Div("📊 Muat data untuk melihat statistik detail", className="placeholder-text")]
    
    try:
        df = pd.DataFrame(data)
        
        # Filter data
        if selected_ports:
            df = df[df['pelabuhan'].isin(selected_ports)]
        if selected_years:
            df = df[df['tahun'].isin(selected_years)]
        
        if df.empty:
            return [html.Div("⚠️ Tidak ada data dengan filter yang dipilih", className="warning-text")]
        
        # Hitung statistik per pelabuhan
        stats_list = []
        
        for pelabuhan in df['pelabuhan'].unique():
            port_data = df[df['pelabuhan'] == pelabuhan]['volume']
            
            stats_card = html.Div([
                html.H5(f"🏴󠁧󠁢󠁥󠁮󠁧󠁿 {pelabuhan}", className="port-stats-title"),
                html.Div([
                    html.Div([
                        html.Strong("📊 Mean: "),
                        html.Span(f"{port_data.mean():,.0f} ton")
                    ], className="stat-item"),
                    html.Div([
                        html.Strong("📈 Median: "),
                        html.Span(f"{port_data.median():,.0f} ton")
                    ], className="stat-item"),
                    html.Div([
                        html.Strong("📏 Std Dev: "),
                        html.Span(f"{port_data.std():,.0f} ton")
                    ], className="stat-item"),
                    html.Div([
                        html.Strong("⬆️ Max: "),
                        html.Span(f"{port_data.max():,.0f} ton")
                    ], className="stat-item"),
                    html.Div([
                        html.Strong("⬇️ Min: "),
                        html.Span(f"{port_data.min():,.0f} ton")
                    ], className="stat-item"),
                    html.Div([
                        html.Strong("📐 CV: "),
                        html.Span(f"{(port_data.std()/port_data.mean()*100):.1f}%")
                    ], className="stat-item"),
                ], className="port-stats-content")
            ], className="port-stats-card")
            
            stats_list.append(stats_card)
        
        return stats_list
    except Exception as e:
        return [html.Div(f"❌ Error: {str(e)}", className="error-text")]

# Callback untuk data table
@app.callback(
    Output('data-table-container', 'children'),
    [Input('data-store', 'data'),
     Input('port-selector', 'value'),
     Input('year-selector', 'value')],
    prevent_initial_call=True
)
def update_data_table(data, selected_ports, selected_years):
    if not data:
        return html.Div("📊 Muat data untuk melihat tabel", className="placeholder-text")
    
    try:
        df = pd.DataFrame(data)
        
        # Filter data
        if selected_ports:
            df = df[df['pelabuhan'].isin(selected_ports)]
        if selected_years:
            df = df[df['tahun'].isin(selected_years)]
        
        if df.empty:
            return html.Div("⚠️ Tidak ada data dengan filter yang dipilih", className="warning-text")
        
        # Buat summary table
        summary_df = df.groupby(['tahun', 'pelabuhan']).agg({
            'volume': ['sum', 'mean', 'count']
        }).round(0)
        
        summary_df.columns = ['Total Volume', 'Rata-rata Volume', 'Jumlah Record']
        summary_df = summary_df.reset_index()
        
        return dash_table.DataTable(
            data=summary_df.to_dict('records'),
            columns=[
                {"name": "📅 Tahun", "id": "tahun", "type": "numeric"},
                {"name": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Pelabuhan", "id": "pelabuhan", "type": "text"},
                {"name": "📦 Total Volume", "id": "Total Volume", "type": "numeric", "format": FormatTemplate.money(0)},
                {"name": "📊 Rata-rata", "id": "Rata-rata Volume", "type": "numeric", "format": FormatTemplate.money(0)},
                {"name": "📋 Records", "id": "Jumlah Record", "type": "numeric"}
            ],
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'fontFamily': 'Segoe UI, sans-serif'
            },
            style_header={
                'backgroundColor': '#1a5fb4',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                }
            ],
            page_size=10,
            sort_action="native",
            filter_action="native"
        )
    except Exception as e:
        return html.Div(f"❌ Error: {str(e)}", className="error-text")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run_server(host='0.0.0.0', port=port, debug=False)
