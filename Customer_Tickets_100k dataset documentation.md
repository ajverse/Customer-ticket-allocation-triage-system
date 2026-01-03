Column Name,Data Type,Description,Constraints / Logic
Ticket_ID,String,Unique identifier for the support ticket.,Format: TKT-100001 to TKT-200000.
Customer_Name,String,Full name of the customer.,Synthetic global names.
Customer_Email,String,Contact email of the customer.,"Matches name pattern (e.g., firstname.lastname@domain)."
Issue_Category,String,The primary topic of the complaint/request.,"Values: Billing, Technical, Account, Fraud, General Inquiry."
Priority_Level,String,Urgency of the ticket.,"Values: Low, Medium, High, Critical."
Submission_Date,Date,Date the ticket was logged.,"Range: Jan 1, 2021 – Dec 31, 2024."
Resolution_Time_Hours,Integer,Time taken to close the ticket (in hours).,Logic: Correlated with Priority. Critical tickets have lower resolution times (1–24h) than Low priority (up to 72h).
Assigned_Agent,String,Name of the support agent handling the case.,6 distinct agents for performance grouping.
Satisfaction_Score,Integer,Post-resolution CSAT rating (1–5).,"Logic: Skewed distribution (Heavy on 4s and 5s, fewer 1s and 2s)."


Analytical Use Cases
Agent Performance: Group by Assigned_Agent to calculate average Resolution_Time_Hours and Satisfaction_Score.

Priority Adherence: Check if Critical tickets are consistently resolved within 24 hours.

Trend Analysis: Pivot Submission_Date (by Month/Year) vs. Issue_Category to find seasonal spike patterns.