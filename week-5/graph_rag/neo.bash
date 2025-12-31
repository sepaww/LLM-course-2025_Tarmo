docker run -d --name neo4j `
  -p 7474:7474 -p 7687:7687 `
  -e NEO4J_AUTH=neo4j/password123 `
  -e NEO4J_PLUGINS='["apoc"]' `
  -e NEO4J_dbms_security_procedures_unrestricted="apoc.*" `
  -e NEO4J_dbms_security_procedures_allowlist="apoc.*" `
  -v neo4j_data:/data `
  neo4j:5
