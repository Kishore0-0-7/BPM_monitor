import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tempfile
from fpdf import FPDF
import base64
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthReportPDF(FPDF):
    """Custom PDF class for health reports"""
    
    def __init__(self, title="Health Report", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'fonts/DejaVuSans-Bold.ttf', uni=True)
        
    def header(self):
        # Logo
        try:
            logo_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'img', 'logo.png')
            if os.path.exists(logo_path):
                self.image(logo_path, 10, 8, 33)
        except Exception as e:
            logger.warning(f"Could not add logo to PDF: {e}")
        
        # Title
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, self.title, 0, 1, 'C')
        self.ln(10)
        
    def footer(self):
        # Page number
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        self.cell(0, 10, 'HealthAssist AI - For informational purposes only', 0, 0, 'R')
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', True)
        self.ln(4)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()
        
    def add_table(self, headers, data):
        # Table headers
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(240, 240, 240)
        
        # Calculate column width based on header length
        col_width = self.w / len(headers)
        
        for header in headers:
            self.cell(col_width, 10, str(header), 1, 0, 'C', True)
        self.ln()
        
        # Table data
        self.set_font('Arial', '', 12)
        self.set_fill_color(255, 255, 255)
        
        for row in data:
            for item in row:
                self.cell(col_width, 10, str(item), 1, 0, 'C')
            self.ln()
            
    def add_plot(self, plot_func, *args, **kwargs):
        """Add a matplotlib plot to the PDF
        
        Args:
            plot_func: Function that takes (*args, **kwargs) and returns a matplotlib figure
        """
        try:
            # Create a temporary file for the plot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                # Generate the plot
                fig = plot_func(*args, **kwargs)
                
                # Save the plot to the temporary file
                fig.savefig(tmp.name, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Add the plot to the PDF
                self.image(tmp.name, x=10, y=None, w=190)
                
                # Delete the temporary file
                os.unlink(tmp.name)
        except Exception as e:
            logger.error(f"Error adding plot to PDF: {e}")
            self.cell(0, 10, "Error: Could not generate plot", 0, 1, 'C')


class ReportGenerator:
    """Generate various health reports"""
    
    def __init__(self, output_dir=None):
        """Initialize report generator
        
        Args:
            output_dir (str, optional): Directory to save reports. Defaults to 'reports'.
        """
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'data', 'reports')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_bpm_report(self, bpm_data, output_format='pdf'):
        """Generate a report for BPM monitoring session
        
        Args:
            bpm_data (dict): BPM session data
            output_format (str, optional): Output format. Defaults to 'pdf'.
            
        Returns:
            str: Path to the generated report file
        """
        # Extract data
        session_id = bpm_data.get('session_id', 'unknown')
        start_time_str = bpm_data.get('start_time', '')
        bpm_values = bpm_data.get('bpm_values', [])
        timestamps = bpm_data.get('timestamps', [])
        summary = bpm_data.get('summary', {})
        
        if not bpm_values:
            return {'error': 'No BPM data available for report generation'}
        
        # Parse start time
        try:
            start_time = datetime.fromisoformat(start_time_str)
            start_time_fmt = start_time.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            start_time_fmt = 'Unknown'
        
        # Generate report filename
        report_filename = f"bpm_report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # For JSON format, just save the data
        if output_format.lower() == 'json':
            json_path = os.path.join(self.output_dir, f"{report_filename}.json")
            with open(json_path, 'w') as f:
                json.dump(bpm_data, f, indent=4)
            return json_path
        
        # For PDF format, generate a PDF report
        elif output_format.lower() == 'pdf':
            pdf_path = os.path.join(self.output_dir, f"{report_filename}.pdf")
            
            try:
                # Create PDF
                pdf = HealthReportPDF(title=f"Heart Rate Monitoring Report")
                pdf.set_author("HealthAssist AI")
                pdf.set_creator("HealthAssist AI Report Generator")
                
                # Session information
                pdf.chapter_title("Session Information")
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, f"Session ID: {session_id}", 0, 1)
                pdf.cell(0, 10, f"Start Time: {start_time_fmt}", 0, 1)
                pdf.cell(0, 10, f"Duration: {summary.get('duration', 0)} seconds", 0, 1)
                pdf.ln(5)
                
                # Summary statistics
                pdf.chapter_title("Summary Statistics")
                stats_data = [
                    ["Average BPM", "Minimum BPM", "Maximum BPM", "Samples"],
                    [
                        str(summary.get('avg_bpm', 0)), 
                        str(summary.get('min_bpm', 0)), 
                        str(summary.get('max_bpm', 0)), 
                        str(summary.get('samples', 0))
                    ]
                ]
                pdf.add_table(stats_data[0], [stats_data[1]])
                pdf.ln(5)
                
                # BPM Chart
                pdf.chapter_title("Heart Rate Chart")
                
                def bpm_chart(bpm_values, timestamps):
                    """Create BPM chart for PDF"""
                    try:
                        # Parse timestamps
                        time_values = [datetime.fromisoformat(ts) for ts in timestamps]
                    except (ValueError, TypeError):
                        time_values = list(range(len(bpm_values)))
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(time_values, bpm_values, color='#1E88E5', linewidth=2)
                    
                    # Add heart rate zones
                    ax.axhspan(0, 60, alpha=0.2, color='#4CAF50', label='Rest')
                    ax.axhspan(60, 100, alpha=0.2, color='#8BC34A', label='Light')
                    ax.axhspan(100, 140, alpha=0.2, color='#FFC107', label='Moderate')
                    ax.axhspan(140, 170, alpha=0.2, color='#FF9800', label='Vigorous')
                    ax.axhspan(170, 200, alpha=0.2, color='#F44336', label='Maximum')
                    
                    # Set chart properties
                    ax.set_xlabel('Time')
                    ax.set_ylabel('BPM')
                    ax.set_title('Heart Rate Over Time')
                    ax.set_ylim(40, 180)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right')
                    
                    # Format x-axis
                    fig.autofmt_xdate()
                    
                    return fig
                
                # Add BPM chart to PDF
                pdf.add_plot(bpm_chart, bpm_values, timestamps)
                pdf.ln(5)
                
                # Zone Distribution
                pdf.chapter_title("Heart Rate Zone Distribution")
                
                def zone_chart(zones):
                    """Create zone distribution chart for PDF"""
                    zone_names = list(zones.keys())
                    zone_values = list(zones.values())
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.bar(zone_names, zone_values, 
                                 color=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336'])
                    
                    # Add percentage labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{height}%', ha='center', va='bottom')
                    
                    ax.set_ylim(0, 100)
                    ax.set_ylabel('Percentage of Time (%)')
                    ax.set_title('Time Spent in Each Heart Rate Zone')
                    
                    return fig
                
                # Add zone chart to PDF
                zones = summary.get('zones', {})
                if zones:
                    pdf.add_plot(zone_chart, zones)
                
                # Health insights
                pdf.add_page()
                pdf.chapter_title("Health Insights")
                
                # Determine predominant zone
                zones = summary.get('zones', {})
                if zones:
                    predominant_zone = max(zones.items(), key=lambda x: x[1])[0]
                    avg_bpm = summary.get('avg_bpm', 0)
                    
                    insights = ""
                    
                    if predominant_zone == 'Rest':
                        insights += "Your heart rate was predominantly in the Rest zone (below 60 BPM). "
                        insights += "This may indicate good cardiovascular health or a naturally low resting heart rate. "
                        insights += "For some individuals, a very low heart rate may indicate a medical condition called bradycardia."
                    elif predominant_zone == 'Light':
                        insights += "Your heart rate was predominantly in the Light zone (60-100 BPM). "
                        insights += "This is the normal resting heart rate range for most adults. "
                        insights += "It indicates your heart is working efficiently at rest."
                    elif predominant_zone == 'Moderate':
                        insights += "Your heart rate was predominantly in the Moderate zone (100-140 BPM). "
                        insights += "This range typically indicates light to moderate physical activity. "
                        insights += "It could also indicate stress or anxiety if measured at rest."
                    elif predominant_zone == 'Vigorous':
                        insights += "Your heart rate was predominantly in the Vigorous zone (140-170 BPM). "
                        insights += "This range typically indicates intense physical activity. "
                        insights += "If this was recorded at rest, it may warrant medical attention."
                    elif predominant_zone == 'Maximum':
                        insights += "Your heart rate was predominantly in the Maximum zone (above 170 BPM). "
                        insights += "This is typically seen only during very intense exercise near maximum effort. "
                        insights += "If this was recorded at rest, immediate medical attention is recommended."
                    
                    pdf.chapter_body(insights)
                    
                    # Add recommendations based on zone
                    pdf.chapter_title("Recommendations")
                    recommendations = ""
                    
                    if predominant_zone == 'Rest' and avg_bpm < 50:
                        recommendations += "• Consider consulting a healthcare provider if experiencing dizziness or fatigue\n"
                        recommendations += "• Monitor your heart rate regularly\n"
                        recommendations += "• Ensure you're well-hydrated and getting adequate nutrition"
                    elif predominant_zone in ['Rest', 'Light']:
                        recommendations += "• Continue maintaining a healthy lifestyle with regular exercise\n"
                        recommendations += "• Stay hydrated and maintain a balanced diet\n"
                        recommendations += "• Consider cardiovascular exercise to strengthen your heart"
                    elif predominant_zone == 'Moderate':
                        recommendations += "• If measured during rest, consider stress reduction techniques\n"
                        recommendations += "• Ensure adequate sleep and hydration\n"
                        recommendations += "• Consider consulting a healthcare provider if heart rate is consistently elevated at rest"
                    elif predominant_zone in ['Vigorous', 'Maximum']:
                        recommendations += "• If measured during rest, consult a healthcare provider soon\n"
                        recommendations += "• Avoid caffeine, alcohol, and other stimulants\n"
                        recommendations += "• Monitor for symptoms like dizziness, shortness of breath, or chest pain"
                    
                    pdf.chapter_body(recommendations)
                
                # Disclaimer
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, "DISCLAIMER", 0, 1, 'C')
                pdf.set_font('Arial', '', 12)
                disclaimer = """This report is generated for informational purposes only and is not intended as a substitute for professional medical advice, diagnosis, or treatment. 

Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read in this report.

The heart rate data presented here may not be clinically accurate and should be interpreted with caution. Many factors can affect heart rate measurements, including device accuracy, proper positioning, and individual health factors.

If you are experiencing chest pain, shortness of breath, dizziness, or other concerning symptoms, please seek immediate medical attention."""
                
                pdf.multi_cell(0, 10, disclaimer)
                
                # Save PDF
                pdf.output(pdf_path)
                
                return pdf_path
                
            except Exception as e:
                logger.error(f"Error generating PDF report: {e}")
                return {'error': f"Failed to generate PDF report: {str(e)}"}
        
        else:
            return {'error': f"Unsupported output format: {output_format}"}
    
    def generate_symptom_report(self, symptom_data, bpm_data=None, output_format='pdf'):
        """Generate a symptom analysis report, optionally combined with BPM data
        
        Args:
            symptom_data (dict): Symptom analysis data
            bpm_data (dict, optional): BPM session data. Defaults to None.
            output_format (str, optional): Output format. Defaults to 'pdf'.
            
        Returns:
            str: Path to the generated report file
        """
        # Extract symptom data
        conditions = symptom_data.get('possible_conditions', [])
        recommendations = symptom_data.get('recommendations', [])
        disclaimer = symptom_data.get('disclaimer', '')
        
        if not conditions:
            return {'error': 'No symptom analysis data available for report generation'}
        
        # Generate report filename
        report_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"health_report_{report_id}"
        
        # For JSON format, just save the data
        if output_format.lower() == 'json':
            report_data = {
                'report_id': report_id,
                'timestamp': datetime.now().isoformat(),
                'symptom_analysis': symptom_data,
                'bpm_data': bpm_data
            }
            
            json_path = os.path.join(self.output_dir, f"{report_filename}.json")
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            return json_path
        
        # For PDF format, generate a PDF report
        elif output_format.lower() == 'pdf':
            pdf_path = os.path.join(self.output_dir, f"{report_filename}.pdf")
            
            try:
                # Create PDF
                pdf = HealthReportPDF(title="Health Assessment Report")
                pdf.set_author("HealthAssist AI")
                pdf.set_creator("HealthAssist AI Report Generator")
                
                # Report information
                pdf.chapter_title("Report Information")
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, f"Report ID: {report_id}", 0, 1)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
                pdf.ln(5)
                
                # Symptom analysis
                pdf.chapter_title("Symptom Analysis")
                
                # Possible conditions table
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, "Possible Conditions", 0, 1)
                
                # Create table data
                condition_headers = ["Condition", "Probability", "Severity"]
                condition_rows = []
                
                for condition in conditions:
                    name = condition.get('name', condition.get('condition', 'Unknown'))
                    
                    # Format probability
                    probability = condition.get('probability', 0)
                    if isinstance(probability, float) and probability <= 1:
                        probability = f"{int(probability * 100)}%"
                    else:
                        probability = f"{probability}%"
                    
                    # Format severity
                    severity = condition.get('severity', 1)
                    severity_str = '⭐' * severity
                    
                    condition_rows.append([name, probability, severity_str])
                
                pdf.add_table(condition_headers, condition_rows)
                pdf.ln(5)
                
                # Recommendations
                if recommendations:
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Recommendations", 0, 1)
                    
                    pdf.set_font('Arial', '', 12)
                    for rec in recommendations:
                        pdf.cell(10, 10, "•", 0, 0)
                        pdf.multi_cell(0, 10, rec)
                    pdf.ln(5)
                
                # BPM data if available
                if bpm_data:
                    pdf.add_page()
                    pdf.chapter_title("Heart Rate Analysis")
                    
                    # Extract BPM data
                    bpm_values = bpm_data.get('bpm_values', [])
                    timestamps = bpm_data.get('timestamps', [])
                    summary = bpm_data.get('summary', {})
                    
                    if bpm_values:
                        # Summary statistics
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, "Heart Rate Statistics", 0, 1)
                        
                        stats_data = [
                            ["Average BPM", "Minimum BPM", "Maximum BPM"],
                            [
                                str(summary.get('avg_bpm', 0)), 
                                str(summary.get('min_bpm', 0)), 
                                str(summary.get('max_bpm', 0))
                            ]
                        ]
                        pdf.add_table(stats_data[0], [stats_data[1]])
                        pdf.ln(5)
                        
                        # BPM Chart
                        def bpm_chart(bpm_values, timestamps):
                            """Create BPM chart for PDF"""
                            try:
                                # Parse timestamps
                                time_values = [datetime.fromisoformat(ts) for ts in timestamps]
                            except (ValueError, TypeError):
                                time_values = list(range(len(bpm_values)))
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(time_values, bpm_values, color='#1E88E5', linewidth=2)
                            
                            # Add heart rate zones
                            ax.axhspan(0, 60, alpha=0.2, color='#4CAF50')
                            ax.axhspan(60, 100, alpha=0.2, color='#8BC34A')
                            ax.axhspan(100, 140, alpha=0.2, color='#FFC107')
                            ax.axhspan(140, 170, alpha=0.2, color='#FF9800')
                            ax.axhspan(170, 200, alpha=0.2, color='#F44336')
                            
                            # Set chart properties
                            ax.set_xlabel('Time')
                            ax.set_ylabel('BPM')
                            ax.set_title('Heart Rate During Analysis')
                            ax.set_ylim(40, 180)
                            ax.grid(True, alpha=0.3)
                            
                            # Format x-axis
                            fig.autofmt_xdate()
                            
                            return fig
                        
                        # Add BPM chart to PDF
                        pdf.add_plot(bpm_chart, bpm_values, timestamps)
                
                # Combined assessment
                if bpm_data and len(conditions) > 0:
                    pdf.add_page()
                    pdf.chapter_title("Combined Health Assessment")
                    
                    # Get top condition and BPM data
                    top_condition = conditions[0]
                    condition_name = top_condition.get('name', top_condition.get('condition', 'Unknown condition'))
                    condition_prob = top_condition.get('probability', 0)
                    
                    if isinstance(condition_prob, float) and condition_prob <= 1:
                        condition_prob = int(condition_prob * 100)
                    
                    avg_bpm = bpm_data.get('summary', {}).get('avg_bpm', 0)
                    
                    # Generate combined insights
                    insights = f"Based on your symptoms, there is a {condition_prob}% probability of {condition_name}. "
                    
                    if avg_bpm < 60:
                        insights += f"Your average heart rate of {avg_bpm} BPM is below the normal range. "
                        if "fever" in str(symptom_data).lower() or "infection" in str(symptom_data).lower():
                            insights += "A low heart rate combined with signs of infection may indicate a serious condition requiring medical attention."
                        else:
                            insights += "This may be your normal resting heart rate or could indicate a condition like bradycardia."
                    elif 60 <= avg_bpm <= 100:
                        insights += f"Your average heart rate of {avg_bpm} BPM is within the normal range. "
                        if "fever" in str(symptom_data).lower() or "infection" in str(symptom_data).lower():
                            insights += "Though your heart rate is normal, your symptoms may indicate an infection."
                        else:
                            insights += "This suggests your cardiovascular system is responding normally."
                    else:
                        insights += f"Your average heart rate of {avg_bpm} BPM is above the normal resting range. "
                        if "fever" in str(symptom_data).lower() or "infection" in str(symptom_data).lower():
                            insights += "An elevated heart rate with signs of infection may indicate your body is fighting an infection."
                        elif "anxiety" in str(symptom_data).lower() or "stress" in str(symptom_data).lower():
                            insights += "This elevated heart rate is consistent with your reported anxiety/stress symptoms."
                        else:
                            insights += "This could be due to stress, medication, caffeine, or another factor."
                    
                    pdf.chapter_body(insights)
                
                # Disclaimer
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, "DISCLAIMER", 0, 1, 'C')
                pdf.set_font('Arial', '', 12)
                
                if disclaimer:
                    pdf.multi_cell(0, 10, disclaimer)
                else:
                    standard_disclaimer = """This report is generated for informational purposes only and is not intended as a substitute for professional medical advice, diagnosis, or treatment. 

Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read in this report.

If you are experiencing chest pain, shortness of breath, dizziness, or other concerning symptoms, please seek immediate medical attention."""
                    pdf.multi_cell(0, 10, standard_disclaimer)
                
                # Save PDF
                pdf.output(pdf_path)
                
                return pdf_path
                
            except Exception as e:
                logger.error(f"Error generating PDF report: {e}")
                return {'error': f"Failed to generate PDF report: {str(e)}"}
        
        else:
            return {'error': f"Unsupported output format: {output_format}"}

def generate_pdf_report(session_data, symptom_results=None, output_path=None, theme_color="#1E88E5", include_charts=True):
    """
    Generate a PDF report with BPM and optional symptom data.
    
    Args:
        session_data (dict): BPM session data
        symptom_results (dict, optional): Symptom analysis results
        output_path (str, optional): Path to save the PDF. If None, a temporary file will be created.
        theme_color (str, optional): Hex color code for report theme. Defaults to "#1E88E5".
        include_charts (bool, optional): Whether to include charts in the report. Defaults to True.
        
    Returns:
        str: Path to the generated PDF file
    """
    # Create a temporary file if no output path provided
    if not output_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            output_path = tmp.name
    
    try:
        # Convert theme color from hex to RGB
        theme_rgb = tuple(int(theme_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Initialize report generator
        generator = ReportGenerator(output_dir=os.path.dirname(output_path))
        
        # If we have both BPM and symptom data, generate a combined report
        if symptom_results:
            # Create a PDF
            pdf = HealthReportPDF(title="Combined Health Report")
            pdf.set_author("HealthAssist AI")
            pdf.set_creator("HealthAssist AI Report Generator")
            
            # Override colors with theme color
            pdf.set_draw_color(theme_rgb[0], theme_rgb[1], theme_rgb[2])
            pdf.set_fill_color(theme_rgb[0], theme_rgb[1], theme_rgb[2], 0.1)
            
            # Add BPM data section
            if session_data:
                # Extract data
                session_id = session_data.get('session_id', 'unknown')
                start_time_str = session_data.get('start_time', '')
                bpm_values = session_data.get('bpm_values', [])
                timestamps = session_data.get('timestamps', [])
                summary = session_data.get('summary', {})
                
                # Parse start time
                try:
                    start_time = datetime.fromisoformat(start_time_str)
                    start_time_fmt = start_time.strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    start_time_fmt = 'Unknown'
                
                # Session information
                pdf.chapter_title("Heart Rate Monitoring Results")
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, f"Session ID: {session_id}", 0, 1)
                pdf.cell(0, 10, f"Start Time: {start_time_fmt}", 0, 1)
                pdf.cell(0, 10, f"Duration: {summary.get('duration', 0)} seconds", 0, 1)
                pdf.ln(5)
                
                # Summary statistics
                pdf.chapter_title("Heart Rate Statistics")
                stats_data = [
                    ["Average BPM", "Minimum BPM", "Maximum BPM", "Samples"],
                    [
                        str(summary.get('avg_bpm', 0)), 
                        str(summary.get('min_bpm', 0)), 
                        str(summary.get('max_bpm', 0)), 
                        str(summary.get('samples', 0))
                    ]
                ]
                pdf.add_table(stats_data[0], [stats_data[1]])
                pdf.ln(5)
                
                # BPM Chart if requested
                if include_charts and bpm_values and timestamps:
                    pdf.chapter_title("Heart Rate Chart")
                    
                    def bpm_chart(bpm_values, timestamps):
                        """Create BPM chart for PDF"""
                        try:
                            # Parse timestamps
                            time_values = [datetime.fromisoformat(ts) for ts in timestamps]
                        except (ValueError, TypeError):
                            time_values = list(range(len(bpm_values)))
                        
                        # Create chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(time_values, bpm_values, color=theme_color, linewidth=2)
                        
                        # Add heart rate zones
                        ax.axhspan(0, 60, alpha=0.2, color='#4CAF50', label='Rest')
                        ax.axhspan(60, 100, alpha=0.2, color='#8BC34A', label='Light')
                        ax.axhspan(100, 140, alpha=0.2, color='#FFC107', label='Moderate')
                        ax.axhspan(140, 170, alpha=0.2, color='#FF9800', label='Vigorous')
                        ax.axhspan(170, 200, alpha=0.2, color='#F44336', label='Maximum')
                        
                        # Set chart properties
                        ax.set_xlabel('Time')
                        ax.set_ylabel('BPM')
                        ax.set_title(f"Heart Rate Session: {session_id}")
                        ax.set_ylim(40, 180)
                        ax.grid(True, alpha=0.3)
                        
                        # Format x-axis labels
                        plt.xticks(rotation=45)
                        fig.tight_layout()
                        
                        return fig
                    
                    pdf.add_plot(bpm_chart, bpm_values, timestamps)
                    pdf.ln(5)
            
            # Add symptom analysis section
            if symptom_results:
                pdf.add_page()
                pdf.chapter_title("Symptom Analysis Results")
                
                # Display analyzed symptoms
                symptoms_str = ", ".join([s for s in session_data.get('symptoms', [])])
                if symptoms_str:
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Analyzed Symptoms:", 0, 1)
                    pdf.set_font('Arial', '', 12)
                    pdf.multi_cell(0, 10, symptoms_str)
                    pdf.ln(5)
                
                # Possible conditions
                conditions = symptom_results.get('possible_conditions', [])
                if conditions:
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Possible Conditions:", 0, 1)
                    
                    # Create table data
                    condition_headers = ["Condition", "Probability", "Severity"]
                    condition_data = []
                    
                    for condition in conditions:
                        condition_name = condition.get('name', condition.get('condition', 'Unknown'))
                        probability = f"{condition.get('probability', 0)}%"
                        severity = "⚠️" * condition.get('severity', 1)
                        condition_data.append([condition_name, probability, severity])
                    
                    pdf.add_table(condition_headers, condition_data)
                    pdf.ln(5)
                
                # Recommendations
                recommendations = symptom_results.get('recommendations', [])
                if recommendations:
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Recommendations:", 0, 1)
                    pdf.set_font('Arial', '', 12)
                    
                    for rec in recommendations:
                        pdf.cell(0, 10, f"• {rec}", 0, 1)
                    
                    pdf.ln(5)
                
                # Disclaimer
                disclaimer = symptom_results.get('disclaimer', '')
                if disclaimer:
                    pdf.set_font('Arial', 'I', 10)
                    pdf.multi_cell(0, 10, disclaimer)
            
            # Generate PDF
            pdf.output(output_path)
            return output_path
            
        else:
            # If we only have BPM data, use the existing generator
            return generator.generate_bpm_report(session_data, output_format='pdf')
    
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        
        # Create a simple error PDF
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, "Error Generating Report", 0, 1, 'C')
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 10, f"An error occurred while generating the report: {str(e)}")
            pdf.output(output_path)
        except:
            # If even the error PDF fails, write a simple text file
            with open(output_path, 'w') as f:
                f.write(f"Error generating report: {str(e)}")
        
        return output_path 