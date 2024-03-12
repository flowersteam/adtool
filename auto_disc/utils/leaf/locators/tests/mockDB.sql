BEGIN;

CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    content BLOB NOT NULL
    );
CREATE TABLE IF NOT EXISTS tree (
    id INTEGER NOT NULL REFERENCES trajectories(id),
    child_id INTEGER REFERENCES trajectories(id)
    );
INSERT INTO trajectories (content) VALUES ('AA=='); 
INSERT INTO trajectories (content) VALUES ('AAA=');
INSERT INTO trajectories (content) VALUES ('AAAA');
INSERT INTO trajectories (content) VALUES ('AAAAAA==');
INSERT INTO trajectories (content) VALUES ('AAAAAAA=');
INSERT INTO trajectories (content) VALUES ('AAAAAA==');
INSERT INTO trajectories (content) VALUES ('AAAAAAAAAAA=');
INSERT INTO tree (id, child_id) VALUES (1, 2);
INSERT INTO tree (id, child_id) VALUES (2, 3);
INSERT INTO tree (id, child_id) VALUES (3, 4);
INSERT INTO tree (id, child_id) VALUES (4, 5);
INSERT INTO tree (id, child_id) VALUES (2, 6);
INSERT INTO tree (id, child_id) VALUES (6, 7);

COMMIT;
