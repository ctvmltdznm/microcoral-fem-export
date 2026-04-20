#!/bin/bash
# Quick script to check mesh size
EXODUS_FILE=${1:-tiny_aragonite_out.e}

echo "Checking $EXODUS_FILE..."
echo ""

# Get coordinate info (rough estimate from first 50 nodes)
echo "=== Approximate mesh bounds ==="
ncdump -v coord "$EXODUS_FILE" 2>/dev/null | grep -A 150 "coord =" | grep "^ " | head -50 | \
awk 'BEGIN {xmin=1e10; xmax=-1e10; ymin=1e10; ymax=-1e10; zmin=1e10; zmax=-1e10}
{
    for(i=1; i<=NF; i++) {
        gsub(/[,;]/, "", $i);
        if ($i ~ /^-?[0-9]/) {
            val = $i + 0;
            if (NR <= 17) { if (val<xmin) xmin=val; if (val>xmax) xmax=val; }
            else if (NR <= 34) { if (val<ymin) ymin=val; if (val>ymax) ymax=val; }
            else if (NR <= 51) { if (val<zmin) zmin=val; if (val>zmax) zmax=val; }
        }
    }
}
END {
    printf "X: %.2f to %.2f\n", xmin, xmax;
    printf "Y: %.2f to %.2f\n", ymin, ymax;
    printf "Z: %.2f to %.2f\n", zmin, zmax;
    printf "\nUse these in your input file:\n\n";
    printf "  [left_boundary]\n";
    printf "    combinatorial_geometry = '\''x < 0.01'\''\n\n";
    printf "  [right_boundary]\n";
    printf "    combinatorial_geometry = '\''x > %.2f'\''\n\n", xmax-0.01;
    printf "  [bottom_boundary]\n";
    printf "    combinatorial_geometry = '\''y < 0.01'\''\n\n";
    printf "  [back_boundary]\n";
    printf "    combinatorial_geometry = '\''z < 0.01'\''\n";
}'

echo ""
echo "=== Existing boundaries ==="
NUM_SS=$(ncdump -h "$EXODUS_FILE" 2>/dev/null | grep "num_side_sets = " | awk '{print $3}' | tr -d ';')
echo "Side sets: $NUM_SS"
if [ "$NUM_SS" -gt 0 ]; then
    ncdump -v ss_names "$EXODUS_FILE" 2>/dev/null | grep '"'
fi
