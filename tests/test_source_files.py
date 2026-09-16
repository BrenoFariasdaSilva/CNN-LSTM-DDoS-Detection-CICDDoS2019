import tempfile
import unittest
from pathlib import Path

from cnn_lstm_ddos_detection.source_files import discover_csv_files


class SourceFilesTest(unittest.TestCase):
    def test_both_days_exclude_nested_csvs_and_order_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "01-12/TFTP.csv",
                "01-12/Syn.csv",
                "01-12/DrDoS_DNS.csv",
                "01-12/Data_Augmentation/Samples/DrDoS_DNS_data_augmented.csv",
                "01-12/Dataset_Description/preprocessing_summary.csv",
                "01-12/Feature_Analysis/PCA/PCA_Results.csv",
                "01-12/Stacking/Cache_Results/cache.csv",
                "03-11/Portmap.csv",
                "03-11/MSSQL.csv",
                "03-11/LDAP.csv",
                "03-11/some_nested_directory/unrelated.csv",
                "unrelated.csv",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Label\n", encoding="utf-8")

            found = [path.relative_to(root).as_posix() for path in discover_csv_files(root)]
            self.assertEqual(found, [
                "01-12/DrDoS_DNS.csv",
                "01-12/Syn.csv",
                "01-12/TFTP.csv",
                "03-11/LDAP.csv",
                "03-11/MSSQL.csv",
                "03-11/Portmap.csv",
            ])


if __name__ == "__main__":
    unittest.main()
