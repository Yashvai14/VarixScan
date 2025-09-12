from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
import numpy as np

class ReportGenerator:
    """Comprehensive report generation for varicose vein analysis"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        self.ensure_output_dir()
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def ensure_output_dir(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.custom_styles = {
            'Title': ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Title'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            ),
            'Heading': ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading1'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.darkblue,
                borderWidth=1,
                borderColor=colors.darkblue,
                borderPadding=5
            ),
            'SubHeading': ParagraphStyle(
                'CustomSubHeading',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceAfter=10,
                textColor=colors.blue
            ),
            'Normal': ParagraphStyle(
                'CustomNormal',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                alignment=TA_JUSTIFY
            ),
            'Recommendation': ParagraphStyle(
                'Recommendation',
                parent=self.styles['Normal'],
                fontSize=11,
                leftIndent=20,
                bulletIndent=10,
                spaceAfter=6
            )
        }
    
    def generate_standard_report(self, patient_data: Dict, analysis_data: Dict, symptoms_data: Dict = None) -> str:
        """Generate a standard analysis report"""
        filename = f"varicose_report_{patient_data['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []
        
        # Header
        story.extend(self._create_header())
        
        # Patient Information
        story.extend(self._create_patient_section(patient_data))
        
        # Analysis Results
        story.extend(self._create_analysis_section(analysis_data))
        
        # Symptoms Section (if available)
        if symptoms_data:
            story.extend(self._create_symptoms_section(symptoms_data))
        
        # Recommendations
        story.extend(self._create_recommendations_section(analysis_data.get('recommendations', [])))
        
        # Educational Information
        story.extend(self._create_education_section())
        
        # Footer
        story.extend(self._create_footer())
        
        doc.build(story)
        return filepath
    
    def generate_comparison_report(self, patient_data: Dict, analyses_data: List[Dict]) -> str:
        """Generate a comparison report showing progression over time"""
        filename = f"comparison_report_{patient_data['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []
        
        # Header
        story.extend(self._create_header())
        
        # Patient Information
        story.extend(self._create_patient_section(patient_data))
        
        # Comparison Analysis
        story.extend(self._create_comparison_section(analyses_data))
        
        # Trend Chart
        chart_path = self._create_progression_chart(analyses_data, patient_data['id'])
        if chart_path:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Progression Chart", self.custom_styles['Heading']))
            story.append(Image(chart_path, width=6*inch, height=4*inch))
        
        # Overall Assessment
        story.extend(self._create_overall_assessment(analyses_data))
        
        # Footer
        story.extend(self._create_footer())
        
        doc.build(story)
        return filepath
    
    def _create_header(self) -> List:
        """Create report header"""
        header = []
        header.append(Paragraph("Varicose Vein Analysis Report", self.custom_styles['Title']))
        header.append(Spacer(1, 20))
        header.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                                self.styles['Normal']))
        header.append(Spacer(1, 30))
        return header
    
    def _create_patient_section(self, patient_data: Dict) -> List:
        """Create patient information section"""
        section = []
        section.append(Paragraph("Patient Information", self.custom_styles['Heading']))
        
        # Patient details table
        patient_table_data = [
            ['Name:', patient_data.get('name', 'N/A')],
            ['Age:', f"{patient_data.get('age', 'N/A')} years"],
            ['Gender:', patient_data.get('gender', 'N/A')],
            ['Patient ID:', str(patient_data.get('id', 'N/A'))],
            ['Report Date:', datetime.now().strftime('%B %d, %Y')]
        ]
        
        patient_table = Table(patient_table_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        section.append(patient_table)
        section.append(Spacer(1, 20))
        return section
    
    def _create_analysis_section(self, analysis_data: Dict) -> List:
        """Create analysis results section"""
        section = []
        section.append(Paragraph("Analysis Results", self.custom_styles['Heading']))
        
        # Main results
        diagnosis = analysis_data.get('diagnosis', 'N/A')
        severity = analysis_data.get('severity', 'N/A')
        confidence = analysis_data.get('confidence', 0)
        
        # Determine color based on severity
        severity_color = colors.green
        if severity == 'Mild':
            severity_color = colors.orange
        elif severity == 'Moderate':
            severity_color = colors.red
        elif severity == 'Severe':
            severity_color = colors.darkred
        
        section.append(Paragraph(f"<b>Diagnosis:</b> {diagnosis}", self.custom_styles['Normal']))
        section.append(Paragraph(f"<b>Severity:</b> <font color='{severity_color}'>{severity}</font>", 
                                self.custom_styles['Normal']))
        section.append(Paragraph(f"<b>Confidence Level:</b> {confidence}%", self.custom_styles['Normal']))
        
        # Detailed metrics table
        if analysis_data.get('detection_count') is not None:
            metrics_data = [
                ['Metric', 'Value', 'Interpretation'],
                ['Detection Count', str(analysis_data.get('detection_count', 0)), 
                 self._interpret_detection_count(analysis_data.get('detection_count', 0))],
                ['Affected Area Ratio', f"{analysis_data.get('affected_area_ratio', 0):.1%}", 
                 self._interpret_area_ratio(analysis_data.get('affected_area_ratio', 0))],
                ['Confidence Score', f"{confidence}%", self._interpret_confidence(confidence)]
            ]
            
            metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            section.append(Spacer(1, 15))
            section.append(Paragraph("Detailed Metrics", self.custom_styles['SubHeading']))
            section.append(metrics_table)
        
        section.append(Spacer(1, 20))
        return section
    
    def _create_symptoms_section(self, symptoms_data: Dict) -> List:
        """Create symptoms analysis section"""
        section = []
        section.append(Paragraph("Symptom Assessment", self.custom_styles['Heading']))
        
        # Symptoms checklist
        symptoms_list = []
        if symptoms_data.get('pain_level', 0) > 0:
            symptoms_list.append(f"Pain Level: {symptoms_data['pain_level']}/10")
        if symptoms_data.get('swelling'):
            symptoms_list.append("Swelling present")
        if symptoms_data.get('cramping'):
            symptoms_list.append("Cramping experienced")
        if symptoms_data.get('itching'):
            symptoms_list.append("Itching reported")
        if symptoms_data.get('burning_sensation'):
            symptoms_list.append("Burning sensation")
        if symptoms_data.get('leg_heaviness'):
            symptoms_list.append("Leg heaviness")
        if symptoms_data.get('skin_discoloration'):
            symptoms_list.append("Skin discoloration")
        
        for symptom in symptoms_list:
            section.append(Paragraph(f"• {symptom}", self.custom_styles['Normal']))
        
        # Risk factors
        risk_factors = []
        if symptoms_data.get('family_history'):
            risk_factors.append("Family history of varicose veins")
        if symptoms_data.get('occupation_standing'):
            risk_factors.append("Occupation requiring prolonged standing")
        if symptoms_data.get('pregnancy_history'):
            risk_factors.append("Previous pregnancy")
        
        if risk_factors:
            section.append(Spacer(1, 10))
            section.append(Paragraph("Risk Factors:", self.custom_styles['SubHeading']))
            for factor in risk_factors:
                section.append(Paragraph(f"• {factor}", self.custom_styles['Normal']))
        
        section.append(Spacer(1, 20))
        return section
    
    def _create_recommendations_section(self, recommendations: List[str]) -> List:
        """Create recommendations section"""
        section = []
        section.append(Paragraph("Recommendations", self.custom_styles['Heading']))
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                section.append(Paragraph(f"{i}. {rec}", self.custom_styles['Recommendation']))
        else:
            section.append(Paragraph("No specific recommendations available.", self.custom_styles['Normal']))
        
        section.append(Spacer(1, 20))
        return section
    
    def _create_education_section(self) -> List:
        """Create educational information section"""
        section = []
        section.append(Paragraph("About Varicose Veins", self.custom_styles['Heading']))
        
        educational_content = [
            "Varicose veins are enlarged, twisted veins that usually appear on the legs and feet. They occur when the valves in the veins don't work properly, causing blood to pool in the veins.",
            "",
            "<b>Common Causes:</b>",
            "• Age - vein walls weaken over time",
            "• Gender - women are more likely to develop varicose veins",
            "• Pregnancy - increased blood volume and hormonal changes",
            "• Family history - genetic predisposition",
            "• Prolonged standing or sitting",
            "• Obesity - extra weight puts pressure on veins",
            "",
            "<b>Prevention Tips:</b>",
            "• Exercise regularly to improve circulation",
            "• Maintain a healthy weight",
            "• Elevate your legs when resting",
            "• Avoid prolonged standing or sitting",
            "• Wear compression stockings if recommended"
        ]
        
        for content in educational_content:
            if content:
                section.append(Paragraph(content, self.custom_styles['Normal']))
            else:
                section.append(Spacer(1, 8))
        
        section.append(Spacer(1, 20))
        return section
    
    def _create_comparison_section(self, analyses_data: List[Dict]) -> List:
        """Create comparison section for multiple analyses"""
        section = []
        section.append(Paragraph("Progression Analysis", self.custom_styles['Heading']))
        
        if len(analyses_data) < 2:
            section.append(Paragraph("Insufficient data for comparison. At least two analyses are required.", 
                                    self.custom_styles['Normal']))
            return section
        
        # Create comparison table
        comparison_data = [['Date', 'Diagnosis', 'Severity', 'Confidence', 'Status']]
        
        for analysis in analyses_data:
            date_str = analysis.get('created_at', datetime.now()).strftime('%Y-%m-%d') if analysis.get('created_at') else 'N/A'
            comparison_data.append([
                date_str,
                analysis.get('diagnosis', 'N/A'),
                analysis.get('severity', 'N/A'),
                f"{analysis.get('confidence', 0)}%",
                self._determine_status(analysis, analyses_data[0] if analyses_data else None)
            ])
        
        comparison_table = Table(comparison_data, colWidths=[1.2*inch, 2*inch, 1*inch, 1*inch, 1.3*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        section.append(comparison_table)
        section.append(Spacer(1, 20))
        return section
    
    def _create_overall_assessment(self, analyses_data: List[Dict]) -> List:
        """Create overall assessment based on multiple analyses"""
        section = []
        section.append(Paragraph("Overall Assessment", self.custom_styles['Heading']))
        
        if len(analyses_data) >= 2:
            latest = analyses_data[0]
            previous = analyses_data[1]
            
            # Determine trend
            trend = self._calculate_trend(latest, previous)
            trend_text = "stable"
            if trend > 0.1:
                trend_text = "improving"
            elif trend < -0.1:
                trend_text = "worsening"
            
            section.append(Paragraph(f"Based on your recent analyses, your condition appears to be {trend_text}.", 
                                    self.custom_styles['Normal']))
            
            if trend_text == "improving":
                section.append(Paragraph("Continue with your current treatment plan and lifestyle modifications.", 
                                        self.custom_styles['Normal']))
            elif trend_text == "worsening":
                section.append(Paragraph("Consider consulting with your healthcare provider for adjusted treatment options.", 
                                        self.custom_styles['Normal']))
        
        section.append(Spacer(1, 20))
        return section
    
    def _create_footer(self) -> List:
        """Create report footer"""
        footer = []
        footer.append(Spacer(1, 30))
        footer.append(Paragraph("Important Disclaimer", self.custom_styles['SubHeading']))
        footer.append(Paragraph(
            "This report is generated by an AI-powered analysis system and should not be considered as a definitive medical diagnosis. "
            "Please consult with a qualified healthcare professional for proper medical evaluation and treatment advice. "
            "The recommendations provided are general guidelines and may not be suitable for all individuals.",
            self.custom_styles['Normal']
        ))
        footer.append(Spacer(1, 20))
        footer.append(Paragraph("© 2024 Varicose Vein Detection System", self.styles['Normal']))
        return footer
    
    def _create_progression_chart(self, analyses_data: List[Dict], patient_id: int) -> str:
        """Create a progression chart showing severity over time"""
        if len(analyses_data) < 2:
            return None
        
        # Prepare data
        dates = []
        severities = []
        confidences = []
        
        severity_map = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
        
        for analysis in reversed(analyses_data):  # Reverse to show chronological order
            if analysis.get('created_at'):
                dates.append(analysis['created_at'])
                severities.append(severity_map.get(analysis.get('severity', 'Normal'), 0))
                confidences.append(analysis.get('confidence', 0))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Severity plot
        plt.subplot(2, 1, 1)
        plt.plot(dates, severities, marker='o', linewidth=2, markersize=8)
        plt.title('Severity Progression Over Time')
        plt.ylabel('Severity Level')
        plt.yticks([0, 1, 2, 3], ['Normal', 'Mild', 'Moderate', 'Severe'])
        plt.grid(True, alpha=0.3)
        
        # Confidence plot
        plt.subplot(2, 1, 2)
        plt.plot(dates, confidences, marker='s', color='green', linewidth=2, markersize=6)
        plt.title('Detection Confidence Over Time')
        plt.ylabel('Confidence (%)')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f"progression_chart_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        chart_path = os.path.join(self.output_dir, chart_filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _interpret_detection_count(self, count: int) -> str:
        """Interpret detection count"""
        if count == 0:
            return "No veins detected"
        elif count <= 3:
            return "Minimal detections"
        elif count <= 7:
            return "Moderate detections"
        else:
            return "Multiple detections"
    
    def _interpret_area_ratio(self, ratio: float) -> str:
        """Interpret affected area ratio"""
        if ratio < 0.05:
            return "Minimal affected area"
        elif ratio < 0.15:
            return "Moderate affected area"
        else:
            return "Extensive affected area"
    
    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence score"""
        if confidence >= 85:
            return "High confidence"
        elif confidence >= 70:
            return "Moderate confidence"
        else:
            return "Low confidence"
    
    def _determine_status(self, current: Dict, previous: Dict = None) -> str:
        """Determine status compared to previous analysis"""
        if not previous:
            return "Baseline"
        
        severity_map = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
        current_severity = severity_map.get(current.get('severity', 'Normal'), 0)
        previous_severity = severity_map.get(previous.get('severity', 'Normal'), 0)
        
        if current_severity > previous_severity:
            return "Progressed"
        elif current_severity < previous_severity:
            return "Improved"
        else:
            return "Stable"
    
    def _calculate_trend(self, latest: Dict, previous: Dict) -> float:
        """Calculate trend between two analyses"""
        severity_map = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
        latest_score = severity_map.get(latest.get('severity', 'Normal'), 0)
        previous_score = severity_map.get(previous.get('severity', 'Normal'), 0)
        
        return (previous_score - latest_score) / 3.0  # Normalize to -1 to 1 scale

# Initialize report generator
report_generator = ReportGenerator()
