-- Supabase Database Setup for Varicose Vein Application
-- Run this script in your Supabase SQL Editor

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create patients table
CREATE TABLE IF NOT EXISTS public.patients (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INTEGER NOT NULL,
    gender VARCHAR(20) NOT NULL,
    phone VARCHAR(20),
    email VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create analyses table
CREATE TABLE IF NOT EXISTS public.analyses (
    id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT REFERENCES public.patients(id) ON DELETE CASCADE,
    image_path TEXT,
    diagnosis TEXT NOT NULL,
    severity VARCHAR(50),
    confidence DECIMAL(5,2),
    detection_count INTEGER DEFAULT 0,
    affected_area_ratio DECIMAL(5,4),
    recommendations JSONB,
    preprocessing_info JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create symptom_records table
CREATE TABLE IF NOT EXISTS public.symptom_records (
    id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT REFERENCES public.patients(id) ON DELETE CASCADE,
    pain_level INTEGER CHECK (pain_level >= 0 AND pain_level <= 10),
    swelling BOOLEAN DEFAULT FALSE,
    cramping BOOLEAN DEFAULT FALSE,
    itching BOOLEAN DEFAULT FALSE,
    burning_sensation BOOLEAN DEFAULT FALSE,
    leg_heaviness BOOLEAN DEFAULT FALSE,
    skin_discoloration BOOLEAN DEFAULT FALSE,
    ulcers BOOLEAN DEFAULT FALSE,
    duration_symptoms TEXT,
    activity_impact INTEGER CHECK (activity_impact >= 0 AND activity_impact <= 10),
    family_history BOOLEAN DEFAULT FALSE,
    occupation_standing BOOLEAN DEFAULT FALSE,
    pregnancy_history BOOLEAN DEFAULT FALSE,
    previous_treatment TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create reports table
CREATE TABLE IF NOT EXISTS public.reports (
    id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT REFERENCES public.patients(id) ON DELETE CASCADE,
    analysis_id BIGINT REFERENCES public.analyses(id) ON DELETE CASCADE,
    report_type VARCHAR(50) DEFAULT 'standard',
    pdf_path TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create appointments table
CREATE TABLE IF NOT EXISTS public.appointments (
    id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT REFERENCES public.patients(id) ON DELETE CASCADE,
    doctor_name VARCHAR(255),
    appointment_type VARCHAR(100) NOT NULL,
    scheduled_date TIMESTAMPTZ NOT NULL,
    notes TEXT,
    status VARCHAR(20) DEFAULT 'scheduled',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create reminders table
CREATE TABLE IF NOT EXISTS public.reminders (
    id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT REFERENCES public.patients(id) ON DELETE CASCADE,
    reminder_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    scheduled_date TIMESTAMPTZ NOT NULL,
    is_sent BOOLEAN DEFAULT FALSE,
    sent_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create risk_assessments table
CREATE TABLE IF NOT EXISTS public.risk_assessments (
    id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT REFERENCES public.patients(id) ON DELETE CASCADE,
    age_factor DECIMAL(5,2),
    bmi_factor DECIMAL(5,2),
    risk_factors JSONB,
    total_score DECIMAL(5,2),
    risk_level VARCHAR(20),
    recommendations JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create chat_messages table
CREATE TABLE IF NOT EXISTS public.chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create wearable_data table
CREATE TABLE IF NOT EXISTS public.wearable_data (
    id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT REFERENCES public.patients(id) ON DELETE CASCADE,
    device_type VARCHAR(100),
    heart_rate INTEGER,
    blood_pressure_systolic INTEGER,
    blood_pressure_diastolic INTEGER,
    steps_count INTEGER,
    activity_level VARCHAR(20),
    sleep_hours DECIMAL(4,2),
    recorded_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_analyses_patient_id ON public.analyses(patient_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON public.analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_symptom_records_patient_id ON public.symptom_records(patient_id);
CREATE INDEX IF NOT EXISTS idx_reports_patient_id ON public.reports(patient_id);
CREATE INDEX IF NOT EXISTS idx_appointments_patient_id ON public.appointments(patient_id);
CREATE INDEX IF NOT EXISTS idx_appointments_scheduled_date ON public.appointments(scheduled_date);
CREATE INDEX IF NOT EXISTS idx_reminders_patient_id ON public.reminders(patient_id);
CREATE INDEX IF NOT EXISTS idx_reminders_scheduled_date ON public.reminders(scheduled_date);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON public.chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON public.chat_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_wearable_data_patient_id ON public.wearable_data(patient_id);
CREATE INDEX IF NOT EXISTS idx_wearable_data_recorded_at ON public.wearable_data(recorded_at);

-- Create updated_at triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_patients_updated_at BEFORE UPDATE ON public.patients FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_analyses_updated_at BEFORE UPDATE ON public.analyses FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_symptom_records_updated_at BEFORE UPDATE ON public.symptom_records FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_reports_updated_at BEFORE UPDATE ON public.reports FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_appointments_updated_at BEFORE UPDATE ON public.appointments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_reminders_updated_at BEFORE UPDATE ON public.reminders FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_risk_assessments_updated_at BEFORE UPDATE ON public.risk_assessments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS) for all tables
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.symptom_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.appointments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reminders ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.risk_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.wearable_data ENABLE ROW LEVEL SECURITY;

-- Create policies to allow all operations for now (you can restrict these later)
CREATE POLICY "Allow all operations on patients" ON public.patients FOR ALL USING (true);
CREATE POLICY "Allow all operations on analyses" ON public.analyses FOR ALL USING (true);
CREATE POLICY "Allow all operations on symptom_records" ON public.symptom_records FOR ALL USING (true);
CREATE POLICY "Allow all operations on reports" ON public.reports FOR ALL USING (true);
CREATE POLICY "Allow all operations on appointments" ON public.appointments FOR ALL USING (true);
CREATE POLICY "Allow all operations on reminders" ON public.reminders FOR ALL USING (true);
CREATE POLICY "Allow all operations on risk_assessments" ON public.risk_assessments FOR ALL USING (true);
CREATE POLICY "Allow all operations on chat_messages" ON public.chat_messages FOR ALL USING (true);
CREATE POLICY "Allow all operations on wearable_data" ON public.wearable_data FOR ALL USING (true);

-- Insert some sample data for testing (optional)
INSERT INTO public.patients (name, age, gender, phone, email) VALUES
('John Doe', 45, 'Male', '+1234567890', 'john.doe@example.com'),
('Sarah Smith', 38, 'Female', '+1234567891', 'sarah.smith@example.com'),
('Mike Johnson', 52, 'Male', '+1234567892', 'mike.johnson@example.com')
ON CONFLICT DO NOTHING;

-- Insert sample analyses (optional)
INSERT INTO public.analyses (patient_id, diagnosis, severity, confidence, created_at) VALUES
(1, 'Mild Varicose Veins', 'Mild', 92.5, NOW() - INTERVAL '2 days'),
(2, 'No Abnormalities Detected', 'Normal', 98.2, NOW() - INTERVAL '1 day'),
(3, 'Moderate Varicose Veins', 'Moderate', 87.8, NOW() - INTERVAL '3 hours')
ON CONFLICT DO NOTHING;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Supabase database setup completed successfully!';
    RAISE NOTICE 'Tables created: patients, analyses, symptom_records, reports, appointments, reminders, risk_assessments, chat_messages, wearable_data';
    RAISE NOTICE 'Sample data inserted for testing.';
END $$;
