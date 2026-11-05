-- Lab spec: keep all models, add client telemetry and request/response capture.
ALTER AI GATEWAY SNOWFLAKE FROM SPECIFICATION $$
schema_version: 1
models:
  - name: '*'
logging:
  enabled: true
  enable_client_telemetry: true
  capture_payload:
    request_response: true
$$;
DESCRIBE AI GATEWAY SNOWFLAKE;
