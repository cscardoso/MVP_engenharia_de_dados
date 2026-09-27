import os
import sys
import shutil
import tempfile
import logging
import subprocess
from pathlib import Path
import zipfile
import glob
import pandas as pd
import requests
import kagglehub


path_kaggle = kagglehub.dataset_download("gnomows/dados-metereologicos-2018-2024-inmet")


#!/usr/bin/env python3
"""
pipeline.py

Pipeline to:
- baixar dataset do Kaggle (gnomows/dados-metereologicos-2018-2024-inmet)
- descompactar e salvar arquivos brutos em data/raw
- filtrar linhas com UF == 'RJ' e salvar em dados/staged

Uso:
    python pipeline.py
"""



# Config
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = DATA_DIR / "raw"
STAGED_DIR = DATA_DIR / "staged"
PROCESSED_DIR = DATA_DIR / "processed"

# Defaults
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "100000"))
TARGET_UF = os.environ.get("TARGET_UF", "RJ")

# Note: do not call ensure_dirs() here before it's defined; run_pipeline() will call it.

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(out_dir: Path = RAW_DIR) -> Path:
        """
        Baixa o dataset do Kaggle e retorna o caminho do arquivo zip baixado.
        """
        logging.info("Baixando dataset do Kaggle...")
        try:
                dataset = "gnomows/dados-metereologicos-2018-2024-inmet"
                kagglehub.dataset_download(dataset, path=out_dir, unzip=False)
                zip_path = out_dir / "dados-metereologicos-2018-2024-inmet.zip"
                if zip_path.exists():
                        logging.info(f"Dataset baixado em {zip_path}")
                        return zip_path
                else:
                        logging.error("Falha ao baixar o dataset: arquivo zip não encontrado após download.")
                        return None
        except Exception as e:
                logging.error(f"Erro ao baixar o dataset: {e}")
                return None

def extract_archive(zip_path: Path, dest: Path):
        logging.info(f"Extraindo {zip_path} para {dest}...")
        try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(dest)
                logging.info("Extração concluída.")
        except zipfile.BadZipFile:
                logging.error(f"O arquivo {zip_path} não é um arquivo zip válido.")
        except Exception as e:
                logging.error(f"Erro ao extrair {zip_path}: {e}")


def copy_raw_files_from_tmp(tmpdir: Path):
        """
        Move ou copia todos os arquivos relevantes do tmpdir para RAW_DIR.
        Mantém subdiretórios planos.
        """
        logging.info(f"Copiando arquivos brutos para {RAW_DIR}...")
        for p in tmpdir.rglob("*"):
                if p.is_file():
                        # p.relative_to(tmpdir) to keep structure optionally; here flatten keeping names
                        dest = RAW_DIR / p.name
                        # Se mesmo nome existir, adiciona sufixo numérico
                        if dest.exists():
                                base = dest.stem
                                suf = dest.suffix
                                i = 1
                                while True:
                                        dest = RAW_DIR / f"{base}_{i}{suf}"
                                        if not dest.exists():
                                                break
                                        i += 1
                        shutil.copy2(p, dest)


def find_csv_files(directory: Path):
        return [Path(p) for p in glob.glob(str(directory / "**" / "*.csv"), recursive=True)]


def detect_encoding_and_read_chunks(path: Path, chunksize=CHUNK_SIZE):
        encodings = ["utf-8", "latin1", "cp1252"]
        for enc in encodings:
                try:
                        for chunk in pd.read_csv(path, chunksize=chunksize, encoding=enc, dtype=str, low_memory=False):
                                yield chunk
                        return
                except Exception:
                        # tentar próximo encoding
                        continue
        # se chegou aqui, lança
        raise ValueError(f"Não foi possível ler {path} com encodings padrões.")


def find_uf_column(columns):
        for c in columns:
                if str(c).strip().upper() == "UF":
                        return c
        # tentar variações comuns
        for c in columns:
                if "uf" == str(c).strip().lower():
                        return c
        return None


def filter_csv_by_uf(input_path: Path, output_path: Path, uf_value=TARGET_UF):
        logging.info(f"Filtrando {input_path.name} por UF == {uf_value}")
        first_write = True
        wrote_any = False
        try:
                chunk_iter = detect_encoding_and_read_chunks(input_path)
        except Exception as e:
                logging.error(f"Erro ao ler {input_path}: {e}")
                return False

        uf_col = None
        for chunk in chunk_iter:
                if uf_col is None:
                        uf_col = find_uf_column(chunk.columns)
                        if uf_col is None:
                                logging.warning(f"Arquivo {input_path} não contém coluna 'UF' (colunas: {list(chunk.columns)[:10]}). Pulando.")
                                return False
                # Garantir tipos string para evitar problemas
                col_series = chunk[uf_col].astype(str).str.strip().str.upper()
                sel = chunk[col_series == uf_value]
                if not sel.empty:
                        mode = "w" if first_write else "a"
                        header = first_write
                        sel.to_csv(output_path, mode=mode, index=False, header=header, encoding="utf-8")
                        first_write = False
                        wrote_any = True
        if wrote_any:
                logging.info(f"Arquivo filtrado salvo em {output_path}")
        else:
                logging.info(f"Nenhuma linha com UF=={uf_value} encontrada em {input_path.name}")
        return wrote_any


def run_pipeline():
        ensure_dirs()
        # Baixar dataset
        zip_path = download_dataset()
        if zip_path is None:
            logging.error("Falha ao baixar o dataset. Encerrando pipeline.")
            return
        # Extrair diretamente para RAW_DIR
        extract_archive(zip_path, RAW_DIR)


        # Agora processar CSVs em RAW_DIR
        csv_files = find_csv_files(RAW_DIR)
        if not csv_files:
                logging.warning(f"Nenhum arquivo CSV encontrado em {RAW_DIR}. Verifique se o download/descompactação ocorreu corretamente.")
                return

        for csv in csv_files:
                out_path = STAGED_DIR / csv.name
                filter_csv_by_uf(csv, out_path, uf_value=TARGET_UF)


if __name__ == "__main__":
        try:
                run_pipeline()
        except KeyboardInterrupt:
                logging.info("Interrompido pelo usuário")
                sys.exit(1)