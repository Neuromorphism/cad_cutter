from brep_engine.step_index import scan_step_file


def test_scan_step_file_detects_schema_and_tessellation_absence(tmp_path):
    sample = tmp_path / "sample.step"
    sample.write_text(
        "\n".join((
            "ISO-10303-21;",
            "HEADER;",
            "FILE_SCHEMA (( 'AUTOMOTIVE_DESIGN' ));",
            "ENDSEC;",
            "DATA;",
            "#1 = ADVANCED_FACE('NONE',(),#2,.T.);",
            "#2 = B_SPLINE_CURVE_WITH_KNOTS('NONE',3,(),.UNSPECIFIED.,.F.,.F.,(),(),.UNSPECIFIED.);",
            "ENDSEC;",
            "END-ISO-10303-21;",
        )),
        encoding="ascii",
    )

    result = scan_step_file(sample)
    assert result.schema == "AUTOMOTIVE_DESIGN"
    assert result.entity_count == 2
    assert result.has_tessellated_representation is False
    assert result.tessellated_entities == {}
