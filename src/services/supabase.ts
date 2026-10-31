// src/services/supabaseClient.ts
import { createClient } from '@supabase/supabase-js';

// Replace with your Supabase Project URL and anon key
const SUPABASE_URL = 'https://ziirfdjjlkkxeylwzrdm.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InppaXJmZGpqbGtreGV5bHd6cmRtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTMzNjk3ODQsImV4cCI6MjEwODk0NTc4NH0.JCo8YnNdnOFc8RZfb-Hj1PtVF1LxDloMUY91lR0IlIc';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);