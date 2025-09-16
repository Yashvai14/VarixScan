"""
FPDF-based PDF report generator for VarixScan
This provides an alternative to ReportLab for PDF generation with fewer dependencies.
"""

import os
import time
from datetime import datetime
from fpdf import FPDF, HTMLMixin
import json
import base64
from PIL import Image
from io import BytesIO

class VarixScanPDF(FPDF, HTMLMixin):
    """Custom PDF class with VarixScan branding and headers"""
    
    def header(self):
        """Add header to each page"""
        # Title
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(44, 62, 80)  # Dark blue
        self.cell(0, 10, 'VarixScan Medical Report', 0, 1, 'C')
        
        # Date
        self.set_font('helvetica', '', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        
        # Line break
        self.ln(5)
        
        # Line
        self.set_draw_color(44, 62, 80)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(10)

    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        self.cell(0, 10, 'VarixScan - AI-Powered Varicose Vein Detection', 0, 0, 'R')

    def add_section_title(self, title):
        """Add a section title"""
        self.set_font('helvetica', 'B', 12)
        self.set_text_color(44, 62, 80)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, 0, 1, 'L', True)
        self.ln(5)

    def add_image_with_caption(self, image_path, caption=""):
        """Add an image with a caption"""
        try:
            # Calculate width to maintain aspect ratio
            img_width = 160  # fixed width
            self.image(image_path, x=(self.w - img_width)/2, w=img_width)
            
            # Add caption
            if caption:
                self.set_font('helvetica', 'I', 10)
                self.set_text_color(100, 100, 100)
                self.ln(5)
                self.cell(0, 5, caption, 0, 1, 'C')
            
            self.ln(10)
        except Exception as e:
            self.set_text_color(200, 0, 0)
            self.set_font('helvetica', '', 10)
            self.cell(0, 10, f"Image could not be loaded: {str(e)}", 0, 1, 'C')
            self.ln(5)

    def add_data_table(self, data, headers=None):
        """Add a data table with optional headers"""
        self.set_font('helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        
        # Determine column width
        col_width = self.w / (len(data[0]) if data else 1)
        
        # Add headers if provided
        if headers:
            self.set_font('helvetica', 'B', 10)
            self.set_fill_color(230, 230, 230)
            for header in headers:
                self.cell(col_width, 7, str(header), 1, 0, 'C', True)
            self.ln()
        
        # Add data rows
        self.set_font('helvetica', '', 10)
        for row in data:
            for item in row:
                self.cell(col_width, 7, str(item), 1, 0, 'C')
            self.ln()
        
        self.ln(5)

class FPDFReportGenerator:
    """FPDF-based report generator for VarixScan"""
    
    def __init__(self):
        """Initialize the report generator"""
        # Create the reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
    
    def generate_standard_report(self, patient_data, analysis_data, symptoms_data=None):
        """Generate a standard patient report"""
        # Initialize PDF
        pdf = VarixScanPDF()
        pdf.add_page()
        pdf.alias_nb_pages()
        
        # Patient information section
        pdf.add_section_title("Patient Information")
        pdf.set_font('helvetica', '', 11)
        pdf.cell(40, 7, "Patient ID:", 0, 0)
        pdf.cell(0, 7, f"{patient_data.get('id', 'N/A')}", 0, 1)
        pdf.cell(40, 7, "Name:", 0, 0)
        pdf.cell(0, 7, f"{patient_data.get('name', 'N/A')}", 0, 1)
        pdf.cell(40, 7, "Age:", 0, 0)
        pdf.cell(0, 7, f"{patient_data.get('age', 'N/A')}", 0, 1)
        pdf.cell(40, 7, "Gender:", 0, 0)
        pdf.cell(0, 7, f"{patient_data.get('gender', 'N/A')}", 0, 1)
        pdf.ln(10)
        
        # Analysis Results section
        pdf.add_section_title("Analysis Results")
        pdf.set_font('helvetica', '', 11)
        pdf.cell(60, 7, "Diagnosis:", 0, 0)
        pdf.cell(0, 7, f"{analysis_data.get('diagnosis', 'N/A')}", 0, 1)
        pdf.cell(60, 7, "Severity:", 0, 0)
        
        # Color-code severity
        severity = analysis_data.get('severity', 'N/A')
        severity_colors = {
            'Normal': (0, 128, 0),  # Green
            'Mild': (255, 191, 0),  # Yellow
            'Moderate': (255, 127, 0),  # Orange
            'Severe': (255, 0, 0)   # Red
        }
        
        color = severity_colors.get(severity, (0, 0, 0))
        pdf.set_text_color(*color)
        pdf.cell(0, 7, f"{severity}", 0, 1)
        pdf.set_text_color(0, 0, 0)  # Reset to black
        
        pdf.cell(60, 7, "Confidence:", 0, 0)
        pdf.cell(0, 7, f"{analysis_data.get('confidence', 'N/A')}%", 0, 1)
        pdf.cell(60, 7, "Detection Count:", 0, 0)
        pdf.cell(0, 7, f"{analysis_data.get('detection_count', 'N/A')}", 0, 1)
        pdf.cell(60, 7, "Affected Area Ratio:", 0, 0)
        pdf.cell(0, 7, f"{analysis_data.get('affected_area_ratio', 'N/A')}", 0, 1)
        pdf.cell(60, 7, "Date of Analysis:", 0, 0)
        
        # Format date if available
        created_at = analysis_data.get('created_at')
        if created_at:
            if isinstance(created_at, str):
                try:
                    date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    formatted_date = created_at
            else:
                formatted_date = str(created_at)
        else:
            formatted_date = 'N/A'
            
        pdf.cell(0, 7, formatted_date, 0, 1)
        pdf.ln(10)
        
        # Recommendations section
        pdf.add_section_title("Recommendations")
        pdf.set_font('helvetica', '', 11)
        
        recommendations = analysis_data.get('recommendations', [])
        if recommendations:
            for recommendation in recommendations:
                pdf.cell(10, 7, "•", 0, 0)
                pdf.multi_cell(0, 7, recommendation)
        else:
            pdf.cell(0, 7, "No recommendations available", 0, 1)
        pdf.ln(10)
        
        # Symptoms section (if available)
        if symptoms_data:
            pdf.add_section_title("Reported Symptoms")
            pdf.set_font('helvetica', '', 11)
            
            # Create a list of symptoms and their values
            symptom_list = []
            for key, value in symptoms_data.items():
                if key not in ['id', 'patient_id', 'created_at']:
                    if isinstance(value, bool):
                        value = "Yes" if value else "No"
                    symptom_list.append([key.replace('_', ' ').capitalize(), value])
            
            # Add the symptom data as a table
            if symptom_list:
                pdf.add_data_table(symptom_list, ["Symptom", "Value"])
            else:
                pdf.cell(0, 7, "No symptom data available", 0, 1)
            pdf.ln(10)
        
        # Image section (if available)
        try:
            image_path = analysis_data.get('image_path')
            if image_path and os.path.exists(image_path):
                pdf.add_section_title("Analysis Image")
                pdf.add_image_with_caption(image_path, "Varicose Vein Analysis Image")
        except Exception as e:
            pdf.set_text_color(200, 0, 0)
            pdf.cell(0, 7, f"Image could not be loaded: {str(e)}", 0, 1)
            pdf.set_text_color(0, 0, 0)
        
        # Generate filename and save the PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/varicose_report_{patient_data.get('id', '0')}_{timestamp}.pdf"
        pdf.output(filename)
        
        return filename

    def generate_comparison_report(self, patient_data, analyses_data):
        """Generate a comparison report of multiple analyses"""
        # Initialize PDF
        pdf = VarixScanPDF()
        pdf.add_page()
        pdf.alias_nb_pages()
        
        # Patient information section
        pdf.add_section_title("Patient Information")
        pdf.set_font('helvetica', '', 11)
        pdf.cell(40, 7, "Patient ID:", 0, 0)
        pdf.cell(0, 7, f"{patient_data.get('id', 'N/A')}", 0, 1)
        pdf.cell(40, 7, "Name:", 0, 0)
        pdf.cell(0, 7, f"{patient_data.get('name', 'N/A')}", 0, 1)
        pdf.ln(10)
        
        # Comparison table section
        pdf.add_section_title("Analysis Comparison")
        
        # Prepare data for the table
        headers = ["Date", "Diagnosis", "Severity", "Confidence"]
        data = []
        
        for analysis in analyses_data:
            # Format date
            created_at = analysis.get('created_at')
            if created_at:
                if isinstance(created_at, str):
                    try:
                        date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        formatted_date = created_at
                else:
                    formatted_date = str(created_at)
            else:
                formatted_date = 'N/A'
            
            # Add row data
            data.append([
                formatted_date,
                analysis.get('diagnosis', 'N/A'),
                analysis.get('severity', 'N/A'),
                f"{analysis.get('confidence', 'N/A')}%"
            ])
        
        # Add the comparison data as a table
        if data:
            pdf.add_data_table(data, headers)
        else:
            pdf.cell(0, 7, "No comparison data available", 0, 1)
        pdf.ln(10)
        
        # Trend Analysis section
        pdf.add_section_title("Trend Analysis")
        pdf.set_font('helvetica', '', 11)
        
        # Simple trend analysis
        if len(analyses_data) >= 2:
            # Get most recent two analyses
            recent = analyses_data[0]
            previous = analyses_data[1]
            
            # Compare severities
            severity_map = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
            recent_severity = severity_map.get(recent.get('severity'), 0)
            previous_severity = severity_map.get(previous.get('severity'), 0)
            
            if recent_severity > previous_severity:
                trend = "Worsening"
                trend_color = (255, 0, 0)  # Red
            elif recent_severity < previous_severity:
                trend = "Improving"
                trend_color = (0, 128, 0)  # Green
            else:
                trend = "Stable"
                trend_color = (0, 0, 128)  # Blue
            
            # Display trend
            pdf.cell(40, 7, "Current Trend:", 0, 0)
            pdf.set_text_color(*trend_color)
            pdf.cell(0, 7, trend, 0, 1)
            pdf.set_text_color(0, 0, 0)  # Reset to black
            
            # Display change details
            pdf.cell(40, 7, "Severity Change:", 0, 0)
            change = recent_severity - previous_severity
            change_text = f"{'+' if change > 0 else ''}{change} levels"
            if change > 0:
                pdf.set_text_color(255, 0, 0)  # Red for worse
            elif change < 0:
                pdf.set_text_color(0, 128, 0)  # Green for better
            pdf.cell(0, 7, change_text, 0, 1)
            pdf.set_text_color(0, 0, 0)  # Reset to black
        else:
            pdf.cell(0, 7, "Insufficient data for trend analysis", 0, 1)
        
        # Generate filename and save the PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/varicose_comparison_{patient_data.get('id', '0')}_{timestamp}.pdf"
        pdf.output(filename)
        
        return filename

# Create a singleton instance
fpdf_report_generator = FPDFReportGenerator()
