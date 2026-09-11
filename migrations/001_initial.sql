CREATE TABLE users(id uuid PRIMARY KEY, email text UNIQUE NOT NULL, password_hash text NOT NULL,
    role text NOT NULL DEFAULT 'user' CHECK(role IN ('user','admin')));
CREATE TABLE series(id uuid PRIMARY KEY, owner_id uuid NOT NULL REFERENCES users(id),
    name text NOT NULL, revision bigint NOT NULL DEFAULT 0, created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE observations(series_id uuid REFERENCES series(id), day date NOT NULL,
    count integer CHECK(count BETWEEN 0 AND 1000000), PRIMARY KEY(series_id,day));
CREATE TABLE forecast_jobs(id uuid PRIMARY KEY, series_id uuid NOT NULL REFERENCES series(id),
    revision bigint NOT NULL, model_version text NOT NULL, model_sha256 text NOT NULL,
    input jsonb NOT NULL, input_sha256 text NOT NULL, status text NOT NULL DEFAULT 'pending'
      CHECK(status IN ('pending','running','ready','failed')),
    generation uuid, attempts integer NOT NULL DEFAULT 0, lease_until timestamptz,
    result jsonb, error text, created_at timestamptz NOT NULL DEFAULT now(), completed_at timestamptz,
    UNIQUE(series_id,revision));
CREATE INDEX forecast_jobs_ready ON forecast_jobs(status,lease_until,created_at);
CREATE TABLE ingestion_requests(series_id uuid REFERENCES series(id), request_key text NOT NULL,
    body_hash text NOT NULL, result jsonb NOT NULL, PRIMARY KEY(series_id,request_key));
CREATE TABLE worker_heartbeats(name text PRIMARY KEY, seen_at timestamptz NOT NULL DEFAULT now());
