-- One row per called pitch, scored by the streaming job.
CREATE TABLE IF NOT EXISTS decisions (
    game_pk            bigint      NOT NULL,
    at_bat_number      int         NOT NULL,
    pitch_number       int         NOT NULL,
    game_date          date        NOT NULL,
    chal_team          text        NOT NULL,   -- the team that could challenge this call
    chal_side          text        NOT NULL,   -- batter (called strike) or fielder (called ball)
    inning             int         NOT NULL,
    tokens_before      int         NOT NULL,   -- challenges held under the optimal policy
    tokens_actual      int         NOT NULL,   -- challenges the team really held
    w_in               double precision NOT NULL,  -- true miss distance from ABS, inches (>0: umpire wrong)
    y_in               double precision NOT NULL,  -- simulated perceived miss
    sigma_in           double precision NOT NULL,
    p_miss             double precision NOT NULL,  -- P(call is wrong | y)
    g_wp               double precision NOT NULL,  -- stake, win probability
    lam                double precision NOT NULL,  -- shadow price of a challenge
    breakeven_p        double precision NOT NULL,
    optimal_challenge  boolean     NOT NULL,
    ump_wrong          boolean     NOT NULL,
    optimal_gain_wp    double precision NOT NULL,
    actual_challenge   boolean     NOT NULL,
    actual_success     boolean     NOT NULL,
    actual_gain_wp     double precision NOT NULL,
    actual_cva_wp      double precision NOT NULL,
    model_version      text        NOT NULL,
    scored_at          timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, at_bat_number, pitch_number)
);
CREATE INDEX IF NOT EXISTS decisions_scored_at ON decisions (scored_at DESC);
CREATE INDEX IF NOT EXISTS decisions_team ON decisions (chal_team);
CREATE INDEX IF NOT EXISTS decisions_date ON decisions (game_date);

-- One row per drift check by the monitor.
CREATE TABLE IF NOT EXISTS drift_checks (
    checked_at        timestamptz NOT NULL DEFAULT now(),
    model_version     text        NOT NULL,
    rows              int         NOT NULL,
    psi_miss_batter   double precision NOT NULL,
    psi_miss_fielder  double precision NOT NULL,
    psi_stake         double precision NOT NULL,
    ece               double precision NOT NULL,
    brier             double precision NOT NULL,
    miss_rate         double precision NOT NULL,
    alert             boolean     NOT NULL
);

-- A read-only login for Grafana.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana') THEN
        CREATE ROLE grafana LOGIN PASSWORD 'grafana';
    END IF;
END $$;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO grafana;
