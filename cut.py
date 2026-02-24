import sys
import os
import argparse

# --- CONFIGURATION: FREECAD PATH ---
possible_paths = [
    r"C:\Program Files\FreeCAD 1.0\bin",
    r"C:\Program Files\FreeCAD 0.21\bin",
    r"/usr/lib/freecad/lib",
    r"/usr/lib/freecad-python3/lib",
    r"/Applications/FreeCAD.app/Contents/Resources/lib"
]

FREECADPATH = None
for path in possible_paths:
    if os.path.exists(path):
        FREECADPATH = path
        break

if FREECADPATH is None:
    print("Error: Could not find FreeCAD path.")
    sys.exit(1)

if sys.platform == "win32":
    os.environ["PATH"] += os.pathsep + FREECADPATH
sys.path.append(FREECADPATH)

try:
    import FreeCAD
    import Part
    import Import
except ImportError as e:
    print(f"Error importing FreeCAD: {e}")
    sys.exit(1)

def ensure_solid(shape):
    """
    If the shape is a Shell, try to convert it to a Solid.
    Booleans often fail on Shells but work on Solids.
    """
    if shape.ShapeType == "Solid":
        return shape
    
    if shape.ShapeType == "Shell":
        try:
            solid = Part.Solid(shape)
            if solid.isValid():
                return solid
        except:
            pass
    return shape

def robust_slice_and_filter(target_shape, cutter_tool, part_name, debug_mode=False):
    """
    Attempts to slice the target using the explicit 2-argument mode.
    Only prints verbose logs if debug_mode is True.
    """
    # 1. Try to convert to Solid first
    work_shape = ensure_solid(target_shape)
    
    # 2. Try the Slice Method (Preferred)
    if hasattr(work_shape, "slice"):
        try:
            # Explicit "Split" mode to satisfy strict API versions
            sliced = work_shape.slice(cutter_tool, "Split")
            
            # Filter pieces
            survivors = []
            fragments = []
            
            # Handle different return types from slice
            if sliced.Solids: fragments.extend(sliced.Solids)
            elif sliced.Shells: fragments.extend(sliced.Shells)
            elif sliced.Faces: fragments.extend(sliced.Faces)
            else: fragments.append(sliced)
            
            for frag in fragments:
                # Check Center of Mass
                # 'True' checks the volume containment
                if not cutter_tool.isInside(frag.CenterOfMass, 1e-5, True):
                    survivors.append(frag)
            
            # --- DEBUG LOGGING ---
            if debug_mode:
                msg = f"  > {part_name}: Found {len(fragments)} fragments."
                if len(survivors) == 0:
                    print(msg + " REMOVED ALL (Fully inside cutter).")
                elif len(survivors) == len(fragments):
                    if len(fragments) > 1:
                        print(msg + " KEPT ALL (Filter didn't trigger).")
                    # else: silent success (cutter didn't touch it)
                else:
                    print(msg + f" Kept {len(survivors)} pieces (Cut successful).")
            # ---------------------

            if not survivors:
                return None # Everything was cut away
            
            return Part.makeCompound(survivors)
            
        except TypeError:
            # Fallback if specific version doesn't accept "Split"
            try:
                sliced = work_shape.slice(cutter_tool)
                if debug_mode: print(f"  > {part_name}: Retry with 1-arg slice succeeded.")
                return sliced 
            except Exception as e:
                if debug_mode: print(f"  > {part_name}: [ERROR] Slice retry failed: {e}")

        except Exception as e:
            if debug_mode: print(f"  > {part_name}: [ERROR] Slice failed: {e}")
    
    # 3. Fallback: Standard Cut
    try:
        cut_res = work_shape.cut(cutter_tool)
        if not cut_res.isNull() and cut_res.isValid():
            if debug_mode: print(f"  > {part_name}: Fallback 'Cut' succeeded.")
            return cut_res
    except:
        pass
        
    if debug_mode: print(f"  > {part_name}: [WARN] All operations failed. Returning original.")
    return target_shape

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .step file")
    parser.add_argument("output", help="Output .step file")
    parser.add_argument("angle", type=float, help="Cut angle")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logs and export cutter object")
    args = parser.parse_args()

    doc = FreeCAD.newDocument("ProcessingDoc")
    
    try:
        print(f"Loading {args.input}...")
        Import.insert(args.input, doc.Name)
        
        valid_objs = []
        global_bbox = FreeCAD.BoundBox()
        
        # 1. Filter Objects
        for obj in doc.Objects:
            if not hasattr(obj, 'Shape') or obj.Shape.isNull(): continue
            label = obj.Label.lower()
            if any(x in label for x in ['axis', 'plane', 'origin']): continue
            
            bb = obj.Shape.BoundBox
            if bb.DiagonalLength < 1e-6 or bb.DiagonalLength > 100000: continue
            
            valid_objs.append(obj)
            global_bbox.add(bb)

        # 2. Create Cutter (Solid Cylinder Sector)
        # Pinned to Y-Axis (x=0, z=0)
        radius = max(abs(global_bbox.XMax), abs(global_bbox.XMin), abs(global_bbox.ZMax), abs(global_bbox.ZMin)) * 2.5
        height = global_bbox.YLength * 3.0
        y_start = global_bbox.YMin - (global_bbox.YLength * 1.0)
        
        print("Constructing Cutter...")
        cutter_tool = Part.makeCylinder(
            radius, 
            height, 
            FreeCAD.Vector(0, y_start, 0), 
            FreeCAD.Vector(0, 1, 0), 
            args.angle
        )

        export_doc = FreeCAD.newDocument("ExportDoc")
        final_features = []
        
        print(f"Processing {len(valid_objs)} objects...")
        
        for obj in valid_objs:
            # Process Solids individually
            sub_shapes = obj.Shape.Solids
            if not sub_shapes: 
                # If no solids, take the whole shape (might be a Shell)
                sub_shapes = [obj.Shape]
            
            for i, shape in enumerate(sub_shapes):
                part_name = f"{obj.Label}_{i}"
                
                # Perform the operation
                # Pass the debug flag down to the function
                result = robust_slice_and_filter(shape, cutter_tool, part_name, args.debug)
                
                if result:
                    # Optional: Clean up faces
                    try:
                        result = result.removeSplitter()
                    except:
                        pass
                        
                    new_obj = export_doc.addObject("Part::Feature", part_name)
                    new_obj.Shape = result
                    final_features.append(new_obj)

        # 3. Handle Debug Output
        if args.debug:
            print("Debug Mode ON: Including Cutter in export.")
            dbg = export_doc.addObject("Part::Feature", "CUTTER_DEBUG")
            dbg.Shape = cutter_tool
            dbg.Visibility = False
            final_features.append(dbg)

        print(f"Exporting to {args.output}...")
        Import.export(final_features, args.output)
        print("Done.")

    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
    finally:
        FreeCAD.closeDocument("ProcessingDoc")
        FreeCAD.closeDocument("ExportDoc")

if __name__ == "__main__":
    main()