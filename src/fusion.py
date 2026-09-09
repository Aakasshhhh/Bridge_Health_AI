def get_bridge_health(image_damage_conf, vib_damage_conf):
    """
    Fuses confidence scores from Image CNN and Vibration LSTM to get a final Health Index.
    """
    combined_damage = (image_damage_conf * 0.6) + (vib_damage_conf * 0.4)

    health_index = (1 - combined_damage) * 100

    if health_index >= 80:
        condition = "Healthy ✅"
    elif health_index >= 50:
        condition = "Needs Monitoring ⚠️"
    else:
        condition = "Critical Condition 🚨"

    return round(health_index, 2), condition