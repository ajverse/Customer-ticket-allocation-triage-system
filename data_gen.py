import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# --- Configuration ---
NUM_ROWS = 20000  # Size of dataset
OUTPUT_FILENAME = "enhanced_customer_support_data.csv"

# --- 1. Define Knowledge Base (The "Brain" of the generator) ---
# This ensures the Text matches the Category (Crucial for ML training)

issue_templates = {
    "Billing": {
        "subjects": ["Invoice discrepancy", "Charged twice", "Refund status", "Payment failed", "Update credit card", "Suspicious charge"],
        "bodies": [
            "I noticed a double charge on my statement for last month.",
            "Why is my bill higher than the agreed amount?",
            "I have been trying to update my payment method but it keeps failing.",
            "I requested a refund 5 days ago, where is it?",
            "Can you send me the invoice for transaction #99283?",
            "My subscription renewed automatically, but I wanted to cancel."
        ]
    },
    "Technical": {
        "subjects": ["Login failed", "App crashing", "API Error 500", "Screen freezes", "Data not syncing", "Installation issue"],
        "bodies": [
            "I cannot log into my account even after resetting the password.",
            "The application crashes every time I open the settings tab.",
            "We are receiving a 500 Internal Server Error on the API endpoint.",
            "The dashboard is not loading any data, just a spinning wheel.",
            "How do I install the latest patch on Windows 11?",
            "My data hasn't synced to the cloud for 24 hours."
        ]
    },
    "Account": {
        "subjects": ["Password reset", "Change email", "Delete account", "2FA issues", "Profile update", "Subscription upgrade"],
        "bodies": [
            "I am not receiving the password reset email.",
            "Please help me change the email address associated with my account.",
            "I want to permanently delete my account and data.",
            "I lost my phone and cannot pass the 2FA check.",
            "How do I upgrade to the Enterprise plan?",
            "My profile picture is not updating."
        ]
    },
    "Fraud": {
        "subjects": ["Unrecognized login", "Stolen card", "Suspicious activity", "Account hacked", "Phishing attempt", "Alert notification"],
        "bodies": [
            "I received a login alert from a device in Russia, which wasn't me.",
            "Someone used my card to make a purchase on your site.",
            "I think my account has been compromised.",
            "I received a suspicious email claiming to be support.",
            "Lock my account immediately, I see transactions I didn't make.",
            "Why is there a new device added to my trusted list?"
        ]
    },
    "General Inquiry": {
        "subjects": ["Product question", "Hours of operation", "Office location", "Feature request", "Pricing tiers", "Demo request"],
        "bodies": [
            "What are your support operating hours?",
            "Do you offer a discount for non-profits?",
            "I would like to request a demo for my team.",
            "Where is your headquarters located?",
            "Is there a roadmap for new features this year?",
            "How does the team pricing tier work?"
        ]
    }
}

# --- 2. Helper Functions ---

def generate_ticket_data(row_id):
    # 1. Randomly pick a category first
    category = np.random.choice(list(issue_templates.keys()), p=[0.25, 0.30, 0.20, 0.05, 0.20]) # Fraud is rare (5%)
    
    # 2. Generate Text based on Category (The "Fix")
    subject_template = random.choice(issue_templates[category]["subjects"])
    body_template = random.choice(issue_templates[category]["bodies"])
    
    # Add noise/variation to text to make it realistic
    ticket_subject = f"{subject_template} - {fake.word().title()}"
    ticket_body = f"Hi Support, {body_template} {fake.sentence()}"

    # 3. Determine Priority (Business Logic)
    # Fraud = Critical/High, General = Low/Medium
    if category == "Fraud":
        priority = np.random.choice(["High", "Critical"], p=[0.3, 0.7])
    elif category == "Technical":
        priority = np.random.choice(["Low", "Medium", "High", "Critical"], p=[0.2, 0.4, 0.3, 0.1])
    else:
        priority = np.random.choice(["Low", "Medium", "High"], p=[0.5, 0.4, 0.1])

    # 4. Determine Resolution Time (Logic: Critical gets fixed faster or takes extremely long if complex)
    if priority == "Critical":
        res_hours = int(np.random.exponential(scale=12)) + 1 # Average 12 hours
    elif priority == "High":
        res_hours = int(np.random.exponential(scale=24)) + 1
    else:
        res_hours = int(np.random.exponential(scale=48)) + 1
    
    # Cap at 72 hours for realism
    res_hours = min(res_hours, 120)

    # 5. Customer Info (Simulate B2B vs B2C)
    is_corporate = random.random() < 0.3 # 30% are corporate
    if is_corporate:
        email_domain = random.choice(["@company.com", "@enterprise.org", "@tech.io"])
        customer_email = f"{fake.first_name()}.{fake.last_name()}{email_domain}"
    else:
        customer_email = fake.email()

    # 6. Satisfaction Score (Skewed: Fraud/Billing issues might leave lower scores)
    if category in ["Billing", "Fraud"]:
        score = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.2, 0.2, 0.2, 0.2]) # Flat distribution (riskier)
    else:
        score = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.05, 0.1, 0.4, 0.4]) # Mostly happy

    return {
        "Ticket_ID": f"TKT-{100000 + row_id}",
        "Customer_Name": fake.name(),
        "Customer_Email": customer_email,
        "Ticket_Subject": ticket_subject,  # NEW COLUMN
        "Ticket_Description": ticket_body, # NEW COLUMN
        "Issue_Category": category,
        "Priority_Level": priority,
        "Ticket_Channel": random.choice(["Email", "Chat", "Web Form"]),
        "Submission_Date": fake.date_between(start_date='-2y', end_date='today'),
        "Resolution_Time_Hours": res_hours,
        "Assigned_Agent": random.choice(["Anya Sharma", "Ben Carter", "Chloe Adams", "David Kim", "Elena Rodriguez"]),
        "Satisfaction_Score": score
    }

# --- 3. Main Loop ---
print(f"Generating {NUM_ROWS} rows of enhanced data...")
data_list = [generate_ticket_data(i) for i in range(NUM_ROWS)]

# --- 4. Convert to DataFrame & Save ---
df = pd.DataFrame(data_list)

# Optional: Add specific "patterns" for advanced ML (e.g., drift)
# E.g., make 'Technical' tickets spike in December
df['Submission_Date'] = pd.to_datetime(df['Submission_Date'])
# (Logic omitted for brevity, but you could boost 'Technical' counts in Dec rows here)

df.to_csv(OUTPUT_FILENAME, index=False)
print(f"Success! Dataset saved as '{OUTPUT_FILENAME}'")
print(df[['Issue_Category', 'Ticket_Subject', 'Priority_Level']].head(5))