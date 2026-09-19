import laya

agent = laya.load("convaiinnovations/laya")
result = agent.predict(
    {"subject": "Duplicate charge on invoice 4411",
     "body": "We were billed twice for March. Please refund the duplicate."},
    {"department": {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "invoices, payments, refunds",
                                 "technical": "bugs and outages", "sales": "pricing"}},
     "urgency": {"type": "score", "instructions": "How urgent is this?",
                 "criteria": ["not urgent", "soon", "blocking"]},
     "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"}},
)
print(result["answers"]["department"]["choice"])
