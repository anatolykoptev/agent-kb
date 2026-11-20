-- kb schema. Derived index over markdown files: drop the database and `kb sync --full` rebuilds it.
-- Requires PostgreSQL 18 (uuidv7) and pgvector.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS kb;

CREATE TABLE IF NOT EXISTS kb.doc (
  doc_id         uuid PRIMARY KEY DEFAULT uuidv7(),
  source         text NOT NULL,              -- the [[source]] name from the config
  layer          text NOT NULL,              -- memory | wiki | raw
  path           text NOT NULL,              -- relative to the source root, NFC
  content_sha256 bytea NOT NULL,
  size           bigint NOT NULL,
  git_sha        text,                       -- HEAD of the wiki repo at index time; null if not a git root
  title          text,
  type           text,
  status         text,
  tags           text[],
  sources        text[],
  doc_date       date,
  frontmatter    jsonb NOT NULL DEFAULT '{}',
  mtime          timestamptz NOT NULL,
  indexed_at     timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source, path)
);

CREATE TABLE IF NOT EXISTS kb.chunk (
  chunk_id     uuid PRIMARY KEY DEFAULT uuidv7(),
  doc_id       uuid NOT NULL REFERENCES kb.doc ON DELETE CASCADE,
  chunk_ix     int  NOT NULL,
  heading_path text NOT NULL DEFAULT '',
  start_line   int  NOT NULL,
  end_line     int  NOT NULL,
  label        text NOT NULL,                -- title + file-name words + heading path (+ description on chunk 0)
  body         text NOT NULL,
  chunk_sha256 bytea NOT NULL,               -- sha256(label || NUL || body): a label change re-embeds
  tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('russian', label), 'A') || setweight(to_tsvector('russian', body), 'D')) STORED,
  embedding    vector(1024),                 -- null = not embedded yet
  embed_model  text,
  UNIQUE (doc_id, chunk_ix)
);
CREATE INDEX IF NOT EXISTS chunk_tsv_gin ON kb.chunk USING gin (tsv);

-- Second full-text vector: no stemming ("simple"), so exact identifiers, names and numbers match as typed.
-- The stoplists are the stock english.stop / russian.stop (an unstopped "simple" matches nearly every chunk on
-- "the", "и", "is"). Latin words use the English list, Cyrillic words the Russian one; numbers, versions, paths
-- and mixed tokens stay unstopped. Idempotent migration: ADD COLUMN IF NOT EXISTS fills the column for every
-- existing row in one pass (CPU only, no re-embedding), and new rows get it from the generation expression.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_ts_dict d JOIN pg_namespace n ON n.oid = d.dictnamespace
                  WHERE n.nspname = 'kb' AND d.dictname = 'simple_en') THEN
    CREATE TEXT SEARCH DICTIONARY kb.simple_en (TEMPLATE = pg_catalog.simple, STOPWORDS = english);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_ts_dict d JOIN pg_namespace n ON n.oid = d.dictnamespace
                  WHERE n.nspname = 'kb' AND d.dictname = 'simple_ru') THEN
    CREATE TEXT SEARCH DICTIONARY kb.simple_ru (TEMPLATE = pg_catalog.simple, STOPWORDS = russian);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_ts_config c JOIN pg_namespace n ON n.oid = c.cfgnamespace
                  WHERE n.nspname = 'kb' AND c.cfgname = 'simple_sw') THEN
    CREATE TEXT SEARCH CONFIGURATION kb.simple_sw (COPY = pg_catalog.simple);
    ALTER TEXT SEARCH CONFIGURATION kb.simple_sw ALTER MAPPING FOR asciiword, asciihword, hword_asciipart WITH kb.simple_en;
    ALTER TEXT SEARCH CONFIGURATION kb.simple_sw ALTER MAPPING FOR word, hword, hword_part WITH kb.simple_ru;
  END IF;
END $$;

ALTER TABLE kb.chunk ADD COLUMN IF NOT EXISTS tsv_simple tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('kb.simple_sw'::regconfig, label), 'A') || setweight(to_tsvector('kb.simple_sw'::regconfig, body), 'D')) STORED;
CREATE INDEX IF NOT EXISTS chunk_tsv_simple_gin ON kb.chunk USING gin (tsv_simple);

CREATE TABLE IF NOT EXISTS kb.search_log (
  at      timestamptz DEFAULT now(),
  session text,
  query   text,
  k       int,
  mode    text,
  hits    int,
  ms      int
);
