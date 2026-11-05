-- Original AI Gateway spec captured 2026-11-04 before the lab (restore with this).
ALTER AI GATEWAY SNOWFLAKE FROM SPECIFICATION $$
schema_version: 1
models:
  - name: '*'
logging:
  enabled: true
$$;
