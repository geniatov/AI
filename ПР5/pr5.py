
# 1. ОПРЕДЕЛЕНИЕ БАЗЫ ЗНАНИЙ (ПРАВИЛА)
# Правила устанавливают требования к СЗИ в зависимости от уровня
# конфиденциальности
KNOWLEDGE_BASE = {
    'level_low': {
        'Mandatory': ['Antivirus', 'Basic_Firewall'],
        'Optional': ['Backup_System']
    },
    'level_medium': {
        'Mandatory': ['Antivirus', 'NextGen_Firewall', 'IDS_Monitor'],
        'Optional': ['Data_Encryption_Storage']
    },
    'level_high': {
        'Mandatory': ['Antivirus', 'NextGen_Firewall', 'IDS_Monitor', 'TwoFA',
                      'System_Audit_Log'],
        'Optional': ['Data_Encryption_Storage', 'Access_Control_System']
    }
}

# 2. ФАКТЫ О СИСТЕМЕ (РАБОЧАЯ ПАМЯТЬ)
# Входные данные, которые экспертная система должна оценить
SYSTEM_FACTS = {
    'data_confidentiality': 'High',            # Уровень обрабатываемой информации
    'is_internet_facing': True,                # Требует защиты от внешних угроз
    'contains_personal_data': True,            # Наличие ПДн
    'required_compliance': ['ISO27001', 'GDPR']  # Требования регуляторов
}


# 3. МЕХАНИЗМ ВЫВОДА (FORWARD CHAINING)
def inference_engine(facts, knowledge_base):
    """Применяет правила для выбора СЗИ на основе фактов."""

    required_szi = set()

    confidentiality = facts.get('data_confidentiality', 'Low').lower()
    is_internet = facts.get('is_internet_facing', False)
    has_pdn = facts.get('contains_personal_data', False)
    compliance = facts.get('required_compliance', [])

    # Правило 1: Определение основного уровня по конфиденциальности данных
    level_key = f'level_{confidentiality}'
    if level_key in knowledge_base:
        # Добавляем обязательные СЗИ, соответствующие уровню
        required_szi.update(knowledge_base[level_key]['Mandatory'])
        # Добавляем необязательные СЗИ (если есть дополнительные требования)
        if is_internet and 'Data_Encryption_Storage' in knowledge_base[level_key]['Optional']:
            required_szi.add('Data_Encryption_Storage')
        if has_pdn and 'Access_Control_System' in knowledge_base[level_key]['Optional']:
            required_szi.add('Access_Control_System')

    # Правило 2: Усиление требований на основе регуляторов (условное усиление)
    if 'GDPR' in compliance and 'TwoFA' in required_szi:
        print("--> ВЫВОД: Усиление безопасности из-за соответствия GDPR.")
        required_szi.add('Data_Masking')

    return sorted(required_szi)


# 4. Запуск и вывод результата
selected_szi = inference_engine(SYSTEM_FACTS, KNOWLEDGE_BASE)

print("\n--- Результат работы Экспертной Системы ---")
print(f"Проанализированная система: Конфиденциальность="
      f"{SYSTEM_FACTS['data_confidentiality']}, "
      f"Интернет-доступ={SYSTEM_FACTS['is_internet_facing']}")
print("Требуемый набор СЗИ (на основе правил):")
for szi in selected_szi:
    print(f"- {szi}")
