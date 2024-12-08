# rasterio
import rasterio
import rasterio.enums
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import numpy as np

class Transformations:

    @staticmethod
    def match_rasters(source_raster_filename: str, template_raster_filename: str, out_filename: str, resampling_method: rasterio.enums.Resampling, dest_dtype, nodata_value = -127.0 ) -> None:
        
        ref_source = rasterio.open(source_raster_filename)
        
        ref_template = rasterio.open(template_raster_filename)
        template_meta = ref_template.meta.copy()
        template_meta.update({
            'dtype' : dest_dtype,
            'nodata' : nodata_value
        })

        out_ref = rasterio.open(out_filename, 'w', **template_meta)
        rasterio.warp.reproject(
            source=rasterio.band(ref_source, 1),
            destination=rasterio.band(out_ref, 1),
            src_transform=ref_source.transform,
            src_crs=ref_source.crs,
            dst_transform=out_ref.transform,
            dst_crs=out_ref.crs,
            resampling=resampling_method
        )
        out_ref.close()        
        return None



    @staticmethod
    def to_crs(source_mtx: np.ndarray, source_meta, source_bounds, dest_meta, resampling_method: rasterio.enums.Resampling):
        
        out_transform, out_width, out_height = calculate_default_transform(
            src_crs=source_meta['crs'],
            dst_crs=dest_meta['crs'],
            width=dest_meta['width'],
            height=dest_meta['height'],    
            left=source_bounds.left,
            bottom=source_bounds.bottom,
            right=source_bounds.right,
            top=source_bounds.top        
        )

        res_mtx = np.zeros((out_height, out_width), dtype=np.float32)

        reproject(
            source=source_mtx,
            destination=res_mtx,
            src_transform=source_meta['transform'],
            src_crs=source_meta['crs'],
            dst_transform=out_transform,
            dst_crs=dest_meta['crs'],
            resampling=resampling_method
        )

        updated_meta = source_meta.copy()
        updated_meta.update({
                'crs': dest_meta['crs'],
                'transform': out_transform,
                'width': out_width,
                'height': out_height
            })

        return res_mtx, updated_meta
