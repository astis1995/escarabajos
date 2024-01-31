USE inventario;
DELETE FROM polls_inventario WHERE CODE < 200;
SHOW VARIABLES LIKE "secure_file_priv";
LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\CICIMA-beetles-general-inventory.txt'
INTO TABLE polls_inventario
FIELDS TERMINATED BY '\t'
IGNORE 1 LINES;