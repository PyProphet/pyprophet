from typing import Optional

import duckdb
from loguru import logger
import pandas as pd

from ..._config import ExportIOConfig
from .._base import BaseReader, BaseWriter
from ..util import (
    load_sqlite_scanner,
)


class SqMassReader(BaseReader):
    """
    Class for reading and processing data from an OpenSWATH workflow sqMass-sqlite XIC based file.
    Extended to support exporting functionality.
    """

    def __init__(self, config: ExportIOConfig):
        super().__init__(config)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def __del__(self):
        """Ensure connection is closed when reader is destroyed"""
        if self._conn is not None:
            self._conn.close()

    def read(self) -> pd.DataFrame:
        """
        Read data from the OpenSWATH workflow sqMass-sqlite based file.
        """
        if self.config.context == "export":
            return self._read_export_data()
        raise NotImplementedError("Only export context is currently supported")

    def _read_export_data(self) -> pd.DataFrame:
        """
        Read data for export based on the configured export format.
        """
        self._initialize_connection()
        self._create_indexes()
        self._create_views()

        if self.config.export_format == "parquet":
            return self._read_for_parquet_export()
        raise ValueError(f"Unsupported export format: {self.config.export_format}")

    def _create_indexes(self) -> None:
        """
        Create necessary indexes in the SQLite files for optimal query performance.
        Uses a temporary SQLite connection to create indexes if they don't exist.
        """
        import sqlite3

        # Indexes for the PQP file
        if hasattr(self.config, "pqp_file") and self.config.pqp_file:
            with sqlite3.connect(self.config.pqp_file) as conn:
                cursor = conn.cursor()

                # Check and create indexes for precursor-related tables
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_precursor_id ON PRECURSOR(ID)
                """)
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_peptide_id ON PEPTIDE(ID)
                """)
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping ON PRECURSOR_PEPTIDE_MAPPING(PRECURSOR_ID, PEPTIDE_ID)
                """)
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transition_precursor_mapping ON TRANSITION_PRECURSOR_MAPPING(PRECURSOR_ID, TRANSITION_ID)
                """)
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION(ID)
                """)

                conn.commit()

        # Indexes for the sqMass file
        with sqlite3.connect(self.config.infile) as conn:
            cursor = conn.cursor()

            # Check and create indexes for chromatogram data
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chromatogram_id ON CHROMATOGRAM(ID)
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_data_chromatogram_id ON DATA(CHROMATOGRAM_ID)
            """)

            conn.commit()

    def _initialize_connection(self) -> None:
        """Initialize DuckDB connection and attach SQLite files"""
        if self._conn is None:
            self._conn = duckdb.connect()
            load_sqlite_scanner(self._conn)
            logger.debug("Initializing DuckDB connection for sqMass export")
            self._conn.execute(f"ATTACH '{self.config.infile}' AS xic (READ_ONLY)")

            if hasattr(self.config, "pqp_file") and self.config.pqp_file:
                logger.debug("Attaching PQP file for sqMass conversion")
                self._conn.execute(
                    f"ATTACH '{self.config.pqp_file}' AS pqp (READ_ONLY)"
                )

    def _create_views(self) -> None:
        """Create necessary views for data extraction"""
        if not hasattr(self.config, "pqp_file") or not self.config.pqp_file:
            raise ValueError(
                "PQP file path not configured - required for sqMass conversion"
            )

        self._create_precursor_transition_view()
        self._create_chromatogram_view()
        self._create_merged_views()

    def _create_precursor_transition_view(self) -> None:
        """Create view for precursor-transition mapping"""
        logger.debug("Creating precursor-transition mapping view")
        self._conn.execute("""
        CREATE OR REPLACE VIEW pqp_view AS
        SELECT
            PRECURSOR.ID AS PRECURSOR_ID,
            TRANSITION.ID AS TRANSITION_ID,
            PEPTIDE.MODIFIED_SEQUENCE,
            PRECURSOR.CHARGE AS PRECURSOR_CHARGE,
            TRANSITION.CHARGE AS PRODUCT_CHARGE,
            TRANSITION.DETECTING AS DETECTING_TRANSITION,
            PRECURSOR.DECOY AS PRECURSOR_DECOY,
            TRANSITION.DECOY AS PRODUCT_DECOY
        FROM pqp.PRECURSOR
        INNER JOIN pqp.PRECURSOR_PEPTIDE_MAPPING 
            ON PRECURSOR.ID = pqp.PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
        INNER JOIN pqp.PEPTIDE 
            ON pqp.PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = pqp.PEPTIDE.ID
        INNER JOIN pqp.TRANSITION_PRECURSOR_MAPPING 
            ON PRECURSOR.ID = pqp.TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID
        INNER JOIN pqp.TRANSITION 
            ON pqp.TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = pqp.TRANSITION.ID
        """)
        logger.opt(raw=True).trace(
            self._conn.execute("SELECT * FROM pqp_view LIMIT 20").df()
        )
        logger.opt(raw=True).trace("\n")

    def _create_chromatogram_view(self) -> None:
        """Create view for chromatogram data"""
        logger.debug("Creating chromatogram data view")
        self._conn.execute("""
        CREATE OR REPLACE VIEW chrom_view AS
        SELECT
            CHROMATOGRAM.NATIVE_ID,
            DATA.COMPRESSION,
            DATA.DATA_TYPE,
            DATA.DATA
        FROM xic.CHROMATOGRAM
        INNER JOIN xic.DATA 
            ON xic.DATA.CHROMATOGRAM_ID = xic.CHROMATOGRAM.ID
        """)

    def _create_merged_views(self) -> None:
        """Create merged and pivoted views"""

        # Create precursor and transition specific views
        logger.debug("Creating split views for precursor and transition data")
        self._conn.execute("""
        CREATE OR REPLACE VIEW chrom_prec_view AS
        SELECT 
            *,
            CAST(REGEXP_EXTRACT(NATIVE_ID, '(\\d+)_Precursor_i\\d+', 1) AS STRING) AS PRECURSOR_ID
        FROM chrom_view
        WHERE REGEXP_MATCHES(NATIVE_ID, '_Precursor_i\\d+')
        """)

        self._conn.execute("""
        CREATE OR REPLACE VIEW chrom_trans_view AS
        SELECT 
            *,
            CAST(NATIVE_ID AS STRING) AS TRANSITION_ID
        FROM chrom_view
        WHERE NOT REGEXP_MATCHES(NATIVE_ID, '_Precursor_i\\d+')
        """)

        # print(self._conn.execute("SELECT * FROM chrom_prec_view").df().head(20).info())

        # print(self._conn.execute("SELECT * FROM chrom_trans_view").df().head(20))

        # Create final merged view
        logger.debug("Creating final merged view for chromatogram data")
        self._conn.execute("""
        CREATE OR REPLACE VIEW chrom_final AS
        WITH prec_meta AS (
            SELECT DISTINCT
                PRECURSOR_ID,
                MODIFIED_SEQUENCE,
                PRECURSOR_CHARGE,
                PRECURSOR_DECOY
            FROM pqp_view
        ),
        chrom_prec_merged AS (
            SELECT
                p.PRECURSOR_ID,
                NULL AS TRANSITION_ID,
                p.MODIFIED_SEQUENCE,
                p.PRECURSOR_CHARGE,
                NULL AS PRODUCT_CHARGE,
                1 AS DETECTING_TRANSITION,
                p.PRECURSOR_DECOY,
                NULL AS PRODUCT_DECOY,
                c.NATIVE_ID,
                c.COMPRESSION,
                c.DATA_TYPE,
                c.DATA
            FROM prec_meta p
            INNER JOIN chrom_prec_view c ON p.PRECURSOR_ID = c.PRECURSOR_ID
        ),
        chrom_trans_merged AS (
            SELECT
                p.PRECURSOR_ID,
                p.TRANSITION_ID,
                p.MODIFIED_SEQUENCE,
                p.PRECURSOR_CHARGE,
                p.PRODUCT_CHARGE,
                p.DETECTING_TRANSITION,
                p.PRECURSOR_DECOY,
                p.PRODUCT_DECOY,
                c.NATIVE_ID,
                c.COMPRESSION,
                c.DATA_TYPE,
                c.DATA
            FROM pqp_view p
            INNER JOIN chrom_trans_view c ON p.TRANSITION_ID = c.TRANSITION_ID
        ),
        chrom_combined AS (
            SELECT * FROM chrom_prec_merged
            UNION ALL
            SELECT * FROM chrom_trans_merged
        )
        SELECT
            PRECURSOR_ID,
            TRANSITION_ID,
            MODIFIED_SEQUENCE,
            PRECURSOR_CHARGE,
            PRODUCT_CHARGE,
            DETECTING_TRANSITION,
            PRECURSOR_DECOY,
            PRODUCT_DECOY,
            NATIVE_ID,
            MAX(CASE WHEN DATA_TYPE = 2 THEN DATA END) AS RT_DATA,
            MAX(CASE WHEN DATA_TYPE = 1 THEN DATA END) AS INTENSITY_DATA,
            MAX(CASE WHEN DATA_TYPE = 2 THEN COMPRESSION END) AS RT_COMPRESSION,
            MAX(CASE WHEN DATA_TYPE = 1 THEN COMPRESSION END) AS INTENSITY_COMPRESSION
        FROM chrom_combined
        GROUP BY
            PRECURSOR_ID,
            TRANSITION_ID,
            MODIFIED_SEQUENCE,
            PRECURSOR_CHARGE,
            PRODUCT_CHARGE,
            DETECTING_TRANSITION,
            PRECURSOR_DECOY,
            PRODUCT_DECOY,
            NATIVE_ID
        ORDER BY PRECURSOR_ID, 
            CASE WHEN TRANSITION_ID IS NULL THEN 0 ELSE 1 END,  -- NULLs first
            TRANSITION_ID  -- Then rest of TRANSITION_ID
        """)

    def _read_for_parquet_export(self) -> pd.DataFrame:
        """Read data in format ready for parquet export"""
        return self._conn.execute("SELECT * FROM chrom_final").df()


class SqMassWriter(BaseWriter):
    """
    Class for writing OpenSWATH results from sqMass files to various formats.
    """

    def __init__(self, config: ExportIOConfig):
        super().__init__(config)

    def save_results(self, result, pi0):
        raise NotImplementedError(
            "SqMassWriter does not implement save_results method. Maybe you meant to use the export method?"
        )

    def export(self) -> None:
        """Main entry point for writing data based on configured format"""
        if self.config.export_format == "parquet":
            self._write_parquet()
        else:
            raise ValueError(
                f"Unsupported sqMass export format: {self.config.export_format}. "
                "Supported formats are 'parquet'."
            )

    def _write_parquet(self) -> None:
        """Handle parquet export based on configuration"""
        if self.config.file_type != "sqmass":
            raise ValueError("Parquet export only supported from sqMass files")
        
        conn = duckdb.connect(":memory:")
        load_sqlite_scanner(conn)

        query = self._build_export_query()

        if self.config.export_format == "parquet":
            return self._execute_copy_query(conn, query, self.config.outfile)
        raise ValueError(f"Unsupported export format: {self.config.export_format}")
    

    def _build_export_query(self) -> str:
        return f"""
        WITH pqp_data AS (
            SELECT
                PRECURSOR.ID AS PRECURSOR_ID,
                TRANSITION.ID AS TRANSITION_ID,
                PEPTIDE.MODIFIED_SEQUENCE,
                PRECURSOR.CHARGE AS PRECURSOR_CHARGE,
                TRANSITION.CHARGE AS PRODUCT_CHARGE,
                TRANSITION.DETECTING AS DETECTING_TRANSITION,
                TRANSITION.ORDINAL AS TRANSITION_ORDINAL,
                TRANSITION.TYPE AS TRANSITION_TYPE,
                PRECURSOR.DECOY AS PRECURSOR_DECOY,
                TRANSITION.DECOY AS PRODUCT_DECOY
            FROM sqlite_scan('{self.config.pqp_file}', 'PRECURSOR') as PRECURSOR
            INNER JOIN sqlite_scan('{self.config.pqp_file}',  'PRECURSOR_PEPTIDE_MAPPING')  as PRECURSOR_PEPTIDE_MAPPING
                ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
            INNER JOIN sqlite_scan('{self.config.pqp_file}', 'PEPTIDE') as PEPTIDE
                ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
            INNER JOIN sqlite_scan('{self.config.pqp_file}', 'TRANSITION_PRECURSOR_MAPPING') as TRANSITION_PRECURSOR_MAPPING
                ON PRECURSOR.ID = TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID
            INNER JOIN sqlite_scan('{self.config.pqp_file}', 'TRANSITION') as TRANSITION
                ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
        ),
        chrom_data AS (
            SELECT
                CHROMATOGRAM.NATIVE_ID,
                DATA.COMPRESSION,
                DATA.DATA_TYPE,
                DATA.DATA
            FROM sqlite_scan('{self.config.infile}', 'CHROMATOGRAM') as CHROMATOGRAM
            INNER JOIN sqlite_scan('{self.config.infile}', 'DATA') as DATA
                ON DATA.CHROMATOGRAM_ID = CHROMATOGRAM.ID
        ),
        prec_meta AS (
            SELECT DISTINCT
                PRECURSOR_ID,
                MODIFIED_SEQUENCE,
                PRECURSOR_CHARGE,
                PRECURSOR_DECOY
            FROM pqp_data
        ),
        chrom_prec_merged AS (
            SELECT
                p.PRECURSOR_ID,
                NULL AS TRANSITION_ID,
                p.MODIFIED_SEQUENCE,
                p.PRECURSOR_CHARGE,
                NULL AS PRODUCT_CHARGE,
                1 AS DETECTING_TRANSITION,
                p.PRECURSOR_DECOY,
                NULL AS PRODUCT_DECOY,
                NULL AS TRANSITION_ORDINAL,
                NULL AS TRANSITION_TYPE,
                c.NATIVE_ID,
                c.COMPRESSION,
                c.DATA_TYPE,
                c.DATA
            FROM prec_meta p
            INNER JOIN chrom_data c 
                ON p.PRECURSOR_ID = CAST(REGEXP_EXTRACT(c.NATIVE_ID, '(\\d+)_Precursor_i\\d+', 1) AS STRING)
            WHERE REGEXP_MATCHES(c.NATIVE_ID, '_Precursor_i\\d+')
        ),
        chrom_trans_merged AS (
            SELECT
                p.PRECURSOR_ID,
                p.TRANSITION_ID,
                p.MODIFIED_SEQUENCE,
                p.PRECURSOR_CHARGE,
                p.PRODUCT_CHARGE,
                p.DETECTING_TRANSITION,
                p.PRECURSOR_DECOY,
                p.PRODUCT_DECOY,
                p.TRANSITION_ORDINAL,
                p.TRANSITION_TYPE,
                c.NATIVE_ID,
                c.COMPRESSION,
                c.DATA_TYPE,
                c.DATA
            FROM pqp_data p
            INNER JOIN chrom_data c ON p.TRANSITION_ID = CAST(c.NATIVE_ID AS STRING)
            WHERE NOT REGEXP_MATCHES(c.NATIVE_ID, '_Precursor_i\\d+')
        ),
        chrom_combined AS (
            SELECT * FROM chrom_prec_merged
            UNION ALL
            SELECT * FROM chrom_trans_merged
        )
        SELECT
            PRECURSOR_ID,
            TRANSITION_ID,
            MODIFIED_SEQUENCE,
            PRECURSOR_CHARGE,
            PRODUCT_CHARGE,
            DETECTING_TRANSITION,
            PRECURSOR_DECOY,
            PRODUCT_DECOY,
            TRANSITION_ORDINAL,
            TRANSITION_TYPE,
            NATIVE_ID,
            MAX(CASE WHEN DATA_TYPE = 2 THEN DATA END) AS RT_DATA,
            MAX(CASE WHEN DATA_TYPE = 1 THEN DATA END) AS INTENSITY_DATA,
            MAX(CASE WHEN DATA_TYPE = 2 THEN COMPRESSION END) AS RT_COMPRESSION,
            MAX(CASE WHEN DATA_TYPE = 1 THEN COMPRESSION END) AS INTENSITY_COMPRESSION
        FROM chrom_combined
        GROUP BY
            PRECURSOR_ID,
            TRANSITION_ID,
            MODIFIED_SEQUENCE,
            PRECURSOR_CHARGE,
            PRODUCT_CHARGE,
            DETECTING_TRANSITION,
            PRECURSOR_DECOY,
            PRODUCT_DECOY,
            TRANSITION_ORDINAL,
            TRANSITION_TYPE,
            NATIVE_ID,
        ORDER BY PRECURSOR_ID, 
            CASE WHEN TRANSITION_ID IS NULL THEN 0 ELSE 1 END,
            TRANSITION_ID
    """