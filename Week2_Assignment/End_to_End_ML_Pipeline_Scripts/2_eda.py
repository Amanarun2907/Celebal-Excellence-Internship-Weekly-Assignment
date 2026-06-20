"""
Week 2 Assignment - Tesla EV Deliveries and Production Data Analysis
Script 2: Exploratory Data Analysis (EDA)
Author: Aman
Date: June 17, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class EDAAnalyzer:
    def __init__(self, filepath):
        """Initialize the EDA Analyzer"""
        self.filepath = filepath
        self.df = None
        
    def load_data(self):
        """Load the cleaned dataset"""
        print("=" * 80)
        print("LOADING CLEANED DATA FOR EDA")
        print("=" * 80)
        
        self.df = pd.read_csv(self.filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        print(f"\n✓ Data loaded successfully!")
        print(f"  - Shape: {self.df.shape}")
        print(f"  - Date Range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        return self.df
    
    def univariate_analysis(self):
        """Perform univariate analysis on key variables"""
        print("\n" + "=" * 80)
        print("UNIVARIATE ANALYSIS")
        print("=" * 80)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Univariate Analysis - Distribution of Key Variables', fontsize=16, fontweight='bold')
        
        numerical_cols = ['Estimated_Deliveries', 'Production_Units', 'Avg_Price_USD', 
                         'Battery_Capacity_kWh', 'Range_km', 'CO2_Saved_tons', 
                         'Charging_Stations', 'Year', 'Month']
        
        for idx, col in enumerate(numerical_cols):
            row = idx // 3
            col_idx = idx % 3
            
            # Histogram with KDE
            axes[row, col_idx].hist(self.df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[row, col_idx].set_title(f'{col}', fontweight='bold')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Frequency')
            axes[row, col_idx].grid(True, alpha=0.3)
            
            # Add mean and median lines
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[row, col_idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            axes[row, col_idx].legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('1_univariate_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Univariate analysis plot saved: 1_univariate_analysis.png")
        plt.close()
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.df[numerical_cols].describe())
    
    def bivariate_analysis(self):
        """Perform bivariate analysis"""
        print("\n" + "=" * 80)
        print("BIVARIATE ANALYSIS")
        print("=" * 80)
        
        # 1. Correlation Matrix
        numerical_cols = ['Estimated_Deliveries', 'Production_Units', 'Avg_Price_USD', 
                         'Battery_Capacity_kWh', 'Range_km', 'CO2_Saved_tons', 
                         'Charging_Stations', 'Year', 'Month']
        
        fig, ax = plt.subplots(figsize=(14, 10))
        correlation_matrix = self.df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix - Tesla EV Data', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('2_correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Correlation matrix saved: 2_correlation_matrix.png")
        plt.close()
        
        # Print top correlations
        print("\nTop Positive Correlations (excluding diagonal):")
        corr_pairs = correlation_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs < 1.0]
        top_positive = corr_pairs.nlargest(10)
        print(top_positive)
        
        # 2. Scatter plots for key relationships
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Bivariate Analysis - Key Relationships', fontsize=16, fontweight='bold')
        
        scatter_pairs = [
            ('Estimated_Deliveries', 'Production_Units'),
            ('Avg_Price_USD', 'Estimated_Deliveries'),
            ('Battery_Capacity_kWh', 'Range_km'),
            ('Charging_Stations', 'Estimated_Deliveries'),
            ('Year', 'Estimated_Deliveries'),
            ('CO2_Saved_tons', 'Estimated_Deliveries')
        ]
        
        for idx, (x_col, y_col) in enumerate(scatter_pairs):
            row = idx // 3
            col_idx = idx % 3
            
            axes[row, col_idx].scatter(self.df[x_col], self.df[y_col], alpha=0.5, s=30)
            axes[row, col_idx].set_xlabel(x_col, fontweight='bold')
            axes[row, col_idx].set_ylabel(y_col, fontweight='bold')
            axes[row, col_idx].set_title(f'{x_col} vs {y_col}')
            
            # Add regression line
            z = np.polyfit(self.df[x_col], self.df[y_col], 1)
            p = np.poly1d(z)
            axes[row, col_idx].plot(self.df[x_col], p(self.df[x_col]), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr = self.df[[x_col, y_col]].corr().iloc[0, 1]
            axes[row, col_idx].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                                   transform=axes[row, col_idx].transAxes,
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                   verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig('3_bivariate_scatter_plots.png', dpi=300, bbox_inches='tight')
        print("✓ Bivariate scatter plots saved: 3_bivariate_scatter_plots.png")
        plt.close()
    
    def categorical_analysis(self):
        """Analyze categorical variables"""
        print("\n" + "=" * 80)
        print("CATEGORICAL ANALYSIS")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Categorical Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distribution by Region
        region_deliveries = self.df.groupby('Region')['Estimated_Deliveries'].sum().sort_values(ascending=False)
        axes[0, 0].bar(region_deliveries.index, region_deliveries.values, color='steelblue', edgecolor='black')
        axes[0, 0].set_title('Total Deliveries by Region', fontweight='bold')
        axes[0, 0].set_xlabel('Region')
        axes[0, 0].set_ylabel('Total Deliveries')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Distribution by Model
        model_deliveries = self.df.groupby('Model')['Estimated_Deliveries'].sum().sort_values(ascending=False)
        axes[0, 1].bar(model_deliveries.index, model_deliveries.values, color='coral', edgecolor='black')
        axes[0, 1].set_title('Total Deliveries by Model', fontweight='bold')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Total Deliveries')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Average Price by Model
        avg_price_model = self.df.groupby('Model')['Avg_Price_USD'].mean().sort_values(ascending=False)
        axes[0, 2].bar(avg_price_model.index, avg_price_model.values, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Average Price by Model', fontweight='bold')
        axes[0, 2].set_xlabel('Model')
        axes[0, 2].set_ylabel('Avg Price (USD)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Box plot - Deliveries by Region
        self.df.boxplot(column='Estimated_Deliveries', by='Region', ax=axes[1, 0])
        axes[1, 0].set_title('Deliveries Distribution by Region', fontweight='bold')
        axes[1, 0].set_xlabel('Region')
        axes[1, 0].set_ylabel('Estimated Deliveries')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45)
        
        # 5. Box plot - Price by Model
        self.df.boxplot(column='Avg_Price_USD', by='Model', ax=axes[1, 1])
        axes[1, 1].set_title('Price Distribution by Model', fontweight='bold')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Avg Price (USD)')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=45)
        
        # 6. Source Type Distribution
        source_counts = self.df['Source_Type'].value_counts()
        axes[1, 2].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', 
                      startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
        axes[1, 2].set_title('Data Source Type Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('4_categorical_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Categorical analysis saved: 4_categorical_analysis.png")
        plt.close()
        
        # Print summaries
        print("\nRegion-wise Summary:")
        print(region_deliveries)
        print("\nModel-wise Summary:")
        print(model_deliveries)
    
    def temporal_analysis(self):
        """Analyze temporal trends"""
        print("\n" + "=" * 80)
        print("TEMPORAL ANALYSIS")
        print("=" * 80)
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        fig.suptitle('Temporal Analysis - Time Series Trends', fontsize=16, fontweight='bold')
        
        # 1. Deliveries over time
        monthly_deliveries = self.df.groupby('Date')['Estimated_Deliveries'].sum().sort_index()
        axes[0, 0].plot(monthly_deliveries.index, monthly_deliveries.values, linewidth=2, color='blue')
        axes[0, 0].set_title('Total Deliveries Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Estimated Deliveries')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Production over time
        monthly_production = self.df.groupby('Date')['Production_Units'].sum().sort_index()
        axes[0, 1].plot(monthly_production.index, monthly_production.values, linewidth=2, color='green')
        axes[0, 1].set_title('Total Production Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Production Units')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Average Price over time
        monthly_price = self.df.groupby('Date')['Avg_Price_USD'].mean().sort_index()
        axes[1, 0].plot(monthly_price.index, monthly_price.values, linewidth=2, color='red')
        axes[1, 0].set_title('Average Price Over Time', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Avg Price (USD)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Yearly comparison
        yearly_data = self.df.groupby('Year').agg({
            'Estimated_Deliveries': 'sum',
            'Production_Units': 'sum'
        })
        x = np.arange(len(yearly_data.index))
        width = 0.35
        axes[1, 1].bar(x - width/2, yearly_data['Estimated_Deliveries'], width, label='Deliveries', color='skyblue')
        axes[1, 1].bar(x + width/2, yearly_data['Production_Units'], width, label='Production', color='lightcoral')
        axes[1, 1].set_title('Yearly Deliveries vs Production', fontweight='bold')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Units')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(yearly_data.index)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Quarterly trends
        quarterly_deliveries = self.df.groupby(['Year', 'Quarter'])['Estimated_Deliveries'].sum()
        axes[2, 0].plot(range(len(quarterly_deliveries)), quarterly_deliveries.values, 
                       marker='o', linewidth=2, markersize=4)
        axes[2, 0].set_title('Quarterly Delivery Trends', fontweight='bold')
        axes[2, 0].set_xlabel('Quarter')
        axes[2, 0].set_ylabel('Estimated Deliveries')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Monthly seasonality
        monthly_avg = self.df.groupby('Month')['Estimated_Deliveries'].mean()
        axes[2, 1].bar(monthly_avg.index, monthly_avg.values, color='purple', alpha=0.7, edgecolor='black')
        axes[2, 1].set_title('Average Deliveries by Month (Seasonality)', fontweight='bold')
        axes[2, 1].set_xlabel('Month')
        axes[2, 1].set_ylabel('Avg Deliveries')
        axes[2, 1].set_xticks(range(1, 13))
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('5_temporal_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Temporal analysis saved: 5_temporal_analysis.png")
        plt.close()
    
    def advanced_insights(self):
        """Generate advanced insights"""
        print("\n" + "=" * 80)
        print("ADVANCED INSIGHTS")
        print("=" * 80)
        
        # 1. Growth Rate Analysis
        print("\n1. Year-over-Year Growth Rate:")
        yearly_deliveries = self.df.groupby('Year')['Estimated_Deliveries'].sum()
        growth_rate = yearly_deliveries.pct_change() * 100
        print(growth_rate)
        
        # 2. Model Performance
        print("\n2. Top Performing Models:")
        model_performance = self.df.groupby('Model').agg({
            'Estimated_Deliveries': 'sum',
            'Avg_Price_USD': 'mean',
            'CO2_Saved_tons': 'sum'
        }).sort_values('Estimated_Deliveries', ascending=False)
        print(model_performance)
        
        # 3. Regional Market Share
        print("\n3. Regional Market Share:")
        region_market_share = (self.df.groupby('Region')['Estimated_Deliveries'].sum() / 
                               self.df['Estimated_Deliveries'].sum() * 100)
        print(region_market_share)
        
        # 4. Efficiency Metrics
        print("\n4. Production Efficiency (Deliveries/Production Ratio):")
        self.df['Efficiency_Ratio'] = self.df['Estimated_Deliveries'] / self.df['Production_Units']
        efficiency_by_model = self.df.groupby('Model')['Efficiency_Ratio'].mean().sort_values(ascending=False)
        print(efficiency_by_model)
    
    def run_eda_pipeline(self):
        """Run the complete EDA pipeline"""
        print("\n" + "=" * 80)
        print("TESLA EV DATA - EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        self.load_data()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.categorical_analysis()
        self.temporal_analysis()
        self.advanced_insights()
        
        print("\n" + "=" * 80)
        print("EDA COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nGenerated Visualizations:")
        print("  1. 1_univariate_analysis.png")
        print("  2. 2_correlation_matrix.png")
        print("  3. 3_bivariate_scatter_plots.png")
        print("  4. 4_categorical_analysis.png")
        print("  5. 5_temporal_analysis.png")

if __name__ == "__main__":
    # Initialize EDA analyzer
    eda = EDAAnalyzer('cleaned_data.csv')
    
    # Run EDA pipeline
    eda.run_eda_pipeline()
    
    print("\n" + "=" * 80)
    print("Next Step: Run '3_feature_engineering.py' for Feature Engineering")
    print("=" * 80)
