-- Python for Everybody Database Handout

-- From https://www.py4e.com/lectures3/Pythonlearn-15-Database-Handout.txt
-- Modified JLA 2023-04-03

-- Download and Install: http://sqlitebrowser.org/

-- Using DB Browser for SQLite:
-- - Create New Database : UsersDB (save in  /Users/ambite/Documents/USC/classes/DSCI-510/DSCI-510-Spring-2023-Ambite/Week13-Databases-SQL/Users.db)
-- - Show sqlite file create in directory
-- - Create table: Users
--   - add name of type text
--   - add email of type text
--     (note the SQL syntax being written in the pane below)
-- - Go to Browse Data tab
--   - insert user with name 'JLA' and email 'ambite@isi.edu' by typing it
-- - Go to Execute SQL tab
--   - insert several users (paste list below)

-- Single Table SQL

CREATE TABLE "Users" ("name" TEXT, "email" TEXT)

INSERT INTO Users (name, email) VALUES ('Chuck', 'csev@umich.edu');
INSERT INTO Users (name, email) VALUES ('Colleen', 'cvl@umich.edu');
INSERT INTO Users (name, email) VALUES ('Ted', 'ted@umich.edu');
INSERT INTO Users (name, email) VALUES ('Sally', 'a1@umich.edu');
INSERT INTO Users (name, email) VALUES ('Ted', 'ted@umich.edu');
INSERT INTO Users (name, email) VALUES ('Kristen', 'kf@umich.edu');

-- Note there is a duplicate Ted!

DELETE FROM Users WHERE email='ted@umich.edu'; -- deletes all
-- maybe want to add it back (just one)
INSERT INTO Users (name, email) VALUES ('Ted', 'ted@umich.edu');

UPDATE Users SET name="Charles" WHERE email='csev@umich.edu';

SELECT * FROM Users;       -- * means all attributes

SELECT email FROM Users;   -- or list the columns (attributes) to retrieve

SELECT * FROM Users WHERE email='csev@umich.edu';  -- list users with given email

SELECT * FROM Users WHERE email like '%isi%';  -- list users with email containing isi
 
SELECT * FROM Users ORDER BY email;  

SELECT * FROM Users ORDER BY name DESC;

SELECT count(*) FROM Users;

SELECT count(*) FROM Users WHERE email like '%isi%';


-- back to slides



-- Multi-Table DB and SQL:

-- - Create New Database : Users (save in  /Users/ambite/Documents/USC/classes/DSCI-510/DSCI-510-Spring-2023-Ambite/Week13-Databases-SQL/Users.db)

CREATE TABLE "Artist" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL UNIQUE, 
    "name" TEXT)

CREATE TABLE "Album" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL UNIQUE, 
    artist_id INTEGER,
    "title" TEXT)

CREATE TABLE "Genre" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL UNIQUE, 
    "name" TEXT)

CREATE TABLE "Track" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL UNIQUE, 
    album_id INTEGER, genre_id INTEGER, len INTEGER, rating INTEGER, 
    "title" TEXT, "count" INTEGER)

-- Only insert artist name, the integer key is created and incremented automatically by the DB
INSERT INTO Artist (name) VALUES ('Led Zepplin') 
INSERT INTO Artist (name) VALUES ('AC/DC')

INSERT INTO Genre (name) VALUES ('Rock') ;
INSERT INTO Genre (name) VALUES ('Metal');

INSERT INTO Album (title, artist_id) VALUES ('Who Made Who', 2);   -- 2 is AC/DC
INSERT INTO Album (title, artist_id) VALUES ('IV', 1);             -- 1 is Led Zeppelin

INSERT INTO Track (title, rating, len, count, album_id, genre_id) 
    VALUES ('Black Dog', 5, 297, 0, 2, 1) ;                        -- IV, Rock
INSERT INTO Track (title, rating, len, count, album_id, genre_id) 
    VALUES ('Stairway', 5, 482, 0, 2, 1) ;                         -- IV, Rock
INSERT INTO Track (title, rating, len, count, album_id, genre_id) 
    VALUES ('About to Rock', 5, 313, 0, 1, 2) ;                    -- Who Made Who, Metal
INSERT INTO Track (title, rating, len, count, album_id, genre_id) 
    VALUES ('Who Made Who', 5, 207, 0, 1, 2) ;                     -- Who Made Who, Metal


SELECT Album.title, Artist.name FROM Album JOIN Artist 
    ON Album.artist_id = Artist.id

SELECT Album.title, Album.artist_id, Artist.id, Artist.name 
    FROM Album JOIN Artist ON Album.artist_id = Artist.id

SELECT Album.title, Album.artist_id, Artist.id, Artist.name    -- alternative syntax for join
FROM Album, Artist
WHERE Album.artist_id = Artist.id


SELECT Track.title, Track.genre_id, Genre.id, Genre.name       -- cross-product == cartesian product
    FROM Track JOIN Genre   

SELECT Track.title, Genre.name FROM Track JOIN Genre 
    ON Track.genre_id = Genre.id

SELECT Track.title, Artist.name, Album.title, Genre.name 
FROM Track JOIN Genre JOIN Album JOIN Artist 
    ON Track.genre_id = Genre.id AND Track.album_id = Album.id 
    AND Album.artist_id = Artist.id
 

SELECT Track.title, Artist.name, Album.title, Genre.name        -- alternative syntax for join
FROM Track, Genre, Album, Artist 
WHERE Track.genre_id = Genre.id AND Track.album_id = Album.id 
      AND Album.artist_id = Artist.id
 
-- back to slides



-- Many-Many Relationship

-- create new DB courses

CREATE TABLE User (
    id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name   TEXT UNIQUE,						-- Enforce unique names, logical key
    email  TEXT
) ;

CREATE TABLE Course (
    id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    title  TEXT UNIQUE						-- Enforce unique names, logical key
) ;

CREATE TABLE Takes (
    user_id     INTEGER,
    course_id   INTEGER,
    role        INTEGER,
    PRIMARY KEY (user_id, course_id)
) ;

INSERT INTO User (name, email) VALUES ('Jane', 'jane@tsugi.org');
INSERT INTO User (name, email) VALUES ('Ed', 'ed@tsugi.org');
INSERT INTO User (name, email) VALUES ('Sue', 'sue@tsugi.org');

INSERT INTO Course (title) VALUES ('Python');
INSERT INTO Course (title) VALUES ('SQL');
INSERT INTO Course (title) VALUES ('PHP');

INSERT INTO Takes (user_id, course_id, role) VALUES (1, 1, 1);
INSERT INTO Takes (user_id, course_id, role) VALUES (2, 1, 0);
INSERT INTO Takes (user_id, course_id, role) VALUES (3, 1, 0);

INSERT INTO Takes (user_id, course_id, role) VALUES (1, 2, 0);
INSERT INTO Takes (user_id, course_id, role) VALUES (2, 2, 1);

INSERT INTO Takes (user_id, course_id, role) VALUES (2, 3, 1);
INSERT INTO Takes (user_id, course_id, role) VALUES (3, 3, 0);

SELECT User.name, Takes.role, Course.title
  FROM User JOIN Takes JOIN Course
  ON Takes.user_id = User.id AND Takes.course_id = Course.id
  ORDER BY Course.title, Takes.role DESC, User.name 


