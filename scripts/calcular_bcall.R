# scripts/calcular_bcall.R

# Configurar el repositorio de Posit Package Manager para binarios precompilados en Ubuntu 24.04 (Noble)
local({
  r <- getOption("repos")
  r["CRAN"] <- "https://packagemanager.posit.co/cran/__linux__/noble/latest"
  options(repos = r)
})

# Crear e indicar a R que use una carpeta local de usuario para las librerías
user_lib <- Sys.getenv("R_LIBS_USER")
dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(user_lib, .libPaths()))

# Instalar remotes si no está instalado (más liviano que devtools)
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", lib = user_lib)
}

# Instalar bcall desde la carpeta local parcheada
if (!requireNamespace("bcall", quietly = TRUE)) {
  install.packages("/tmp/bcall_pkg", repos = NULL, type = "source", lib = user_lib)
}

library(bcall)

# Crear directorio data si no existe
dir.create("data/bcall_data_chile", showWarnings = FALSE, recursive = TRUE)

# URLs de los datos
rollcall_url <- "https://alcatruz.github.io/bcall-example/data/CHL-rollcall-2025.csv"
clustering_url <- "https://alcatruz.github.io/bcall-example/data/CHL-clustering-2025-CORREGIDO.csv"

rollcall_path <- "data/bcall_data_chile/CHL-rollcall-2025.csv"
clustering_path <- "data/bcall_data_chile/CHL-clustering-2025-CORREGIDO.csv"

# Descargar archivos
cat("Descargando dataset de votaciones (rollcall)...\n")
if (!file.exists(rollcall_path)) {
  download.file(rollcall_url, destfile = rollcall_path, mode = "wb")
}

cat("Descargando dataset de clustering corregido...\n")
if (!file.exists(clustering_path)) {
  download.file(clustering_url, destfile = clustering_path, mode = "wb")
}

# Cargar datos
cat("Cargando datos en memoria...\n")
rollcall <- read.csv(rollcall_path, row.names = 1)
clustering_corregido <- read.csv(clustering_path, row.names = 1)

# Ejecutar B-Call
cat("Ejecutando B-Call...\n")
resultado_bcall <- bcall(
  rollcall = rollcall,
  clustering = clustering_corregido,
  pivot = "Alessandri_Vergara_Jorge",
  threshold = 0.1,
  verbose = TRUE
)

# Guardar resultados
output_path <- "data/bcall_data_chile/CHL-bcall-scores-2025.csv"
write.csv(resultado_bcall$results, output_path, row.names = FALSE)
cat("¡Éxito! Puntajes resultantes de B-Call calculados y guardados en:", output_path, "\n")
