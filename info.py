# Site-specific technical & financial assumptions
SITE_PROFILES = {
    "Retired Coal Plant (Brownfield)": {
        "capex_mod": 0.88,        # -12% due to existing grid/water
        "delay_prob": 0.8,       # Low (site is pre-industrialized)
        "const_std_mod": 0.75,    # -25% (higher predictability)
        "pue_offset": 0.02,       # Minor PUE impact
        "licensing_mult": 0.75,    # Faster due to existing industrial zoning
        "site_con_mult": 0.8           # Minimal new permits needed
    },
    "Coastal Industrial Zone": {
        "capex_mod": 0.95,        # -5% due to direct seawater cooling
        "delay_prob": 1,       # Medium
        "const_std_mod": 1.0,     # Standard
        "pue_offset": 0.0,        # Standard PUE
        "licensing_mult": 1,
        "site_con_mult": 1.05         # Mid-range: CRZ clearances + intake system builds
    },
    "Inland Greenfield": {
        "capex_mod": 1.15,        # +15% for cooling towers & new grid tie-in
        "delay_prob": 1.5,       # High (new environmental permits)
        "const_std_mod": 1.3,     # +30% uncertainty (geological unknowns)
        "pue_offset": 0.10,       # Higher PUE due to dry cooling
        "licensing_mult": 1.3,
        "site_con_mult": 1.2         # Longest: New land + full env. cycle
    }
}

# --- Macro-Geography: State Level Impacts ---
STATE_PROFILES = {
    "Maharashtra": {"reg_ease": 0.85, "pue_mod": 1.20, "grid_tariff": 8.5, "subsidy_crore": 200, 'cmt_factor': 1},
    "Gujarat": {"reg_ease": 0.95, "pue_mod": 1.22, "grid_tariff": 7.8, "subsidy_crore": 250, 'cmt_factor': 0.85},
    "Tamil Nadu": {"reg_ease": 0.90, "pue_mod": 1.18, "grid_tariff": 8.0, "subsidy_crore": 180, 'cmt_factor': 1.1},
    "Telangana": {"reg_ease": 0.88, "pue_mod": 1.25, "grid_tariff": 8.2, "subsidy_crore": 150, 'cmt_factor': 1.15}
}