# kg-rag
KG-RAG Backend

## How to Setup
1. Clone this repository with `git clone https://github.com/WalkerRout/kg-rag`
2. Navigate into the repository with `cd kg-rag`

## How to Run
1. Ensure docker daemon is running - start docker desktop
2. Start the docker containers with `docker compose up --build`
   - docker daemon should be running before this step
   - if the neo4j container gives a warning like 'chown: changing ownership of '/data/blahblah': Operation not permitted', see the [Bugs](#Bugs) section for fix, then try this step again
3. Open `http://localhost:3000` in your browser to view the frontend. The backend api is exposed to localhost at port `8504`

#### Bugs
- Neo4j/Postgres need permissions for the .gitignore in the /data directory, must be given with
  - `sudo chown 1000:1000 ./neo4j/data/.gitignore` (replacing neo4j with postgres)
