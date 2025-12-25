# safety/health_rules.py
def check_safety(soc, soh, temp):
    alerts = []

    if soc < 15:
        alerts.append("🔋 Low SOC – Recharge soon")

    if soh < 70:
        alerts.append("❤️ Battery health degrading")

    if temp > 50:
        alerts.append("🔥 Battery overheating")

    if not alerts:
        alerts.append("✅ Battery operating normally")

    return alerts
