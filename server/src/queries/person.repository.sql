-- NOTE: This file is auto generated by ./sql-generator

-- PersonRepository.reassignFaces
update "asset_faces"
set
  "personId" = $1
where
  "asset_faces"."personId" = $2

-- PersonRepository.unassignFaces
update "asset_faces"
set
  "personId" = $1
where
  "asset_faces"."sourceType" = $2

-- PersonRepository.delete
delete from "person"
where
  "person"."id" in ($1)

-- PersonRepository.deleteFaces
delete from "asset_faces"
where
  "asset_faces"."sourceType" = $1

-- PersonRepository.getAllWithoutFaces
select
  "person".*
from
  "person"
  left join "asset_faces" on "asset_faces"."personId" = "person"."id"
where
  "asset_faces"."deletedAt" is null
group by
  "person"."id"
having
  count("asset_faces"."assetId") = $1

-- PersonRepository.getFaces
select
  "asset_faces".*,
  (
    select
      to_json(obj)
    from
      (
        select
          "person".*
        from
          "person"
        where
          "person"."id" = "asset_faces"."personId"
      ) as obj
  ) as "person"
from
  "asset_faces"
where
  "asset_faces"."assetId" = $1
  and "asset_faces"."deletedAt" is null
order by
  "asset_faces"."boundingBoxX1" asc

-- PersonRepository.getFaceById
select
  "asset_faces".*,
  (
    select
      to_json(obj)
    from
      (
        select
          "person".*
        from
          "person"
        where
          "person"."id" = "asset_faces"."personId"
      ) as obj
  ) as "person"
from
  "asset_faces"
where
  "asset_faces"."id" = $1
  and "asset_faces"."deletedAt" is null

-- PersonRepository.getFaceByIdWithAssets
select
  "asset_faces".*,
  (
    select
      to_json(obj)
    from
      (
        select
          "person".*
        from
          "person"
        where
          "person"."id" = "asset_faces"."personId"
      ) as obj
  ) as "person",
  (
    select
      to_json(obj)
    from
      (
        select
          "assets".*
        from
          "assets"
        where
          "assets"."id" = "asset_faces"."assetId"
      ) as obj
  ) as "asset"
from
  "asset_faces"
where
  "asset_faces"."id" = $1
  and "asset_faces"."deletedAt" is null

-- PersonRepository.reassignFace
update "asset_faces"
set
  "personId" = $1
where
  "asset_faces"."id" = $2

-- PersonRepository.getByName
select
  "person".*
from
  "person"
where
  (
    "person"."ownerId" = $1
    and (
      lower("person"."name") like $2
      or lower("person"."name") like $3
    )
  )
limit
  $4

-- PersonRepository.getDistinctNames
select distinct
  on (lower("person"."name")) "person"."id",
  "person"."name"
from
  "person"
where
  (
    "person"."ownerId" = $1
    and "person"."name" != $2
  )

-- PersonRepository.getStatistics
select
  count(distinct ("assets"."id")) as "count"
from
  "asset_faces"
  left join "assets" on "assets"."id" = "asset_faces"."assetId"
  and "asset_faces"."personId" = $1
  and "assets"."isArchived" = $2
  and "assets"."deletedAt" is null
where
  "asset_faces"."deletedAt" is null

-- PersonRepository.getNumberOfPeople
select
  count(distinct ("person"."id")) as "total",
  count(distinct ("person"."id")) filter (
    where
      "person"."isHidden" = $1
  ) as "hidden"
from
  "person"
  inner join "asset_faces" on "asset_faces"."personId" = "person"."id"
  inner join "assets" on "assets"."id" = "asset_faces"."assetId"
  and "assets"."deletedAt" is null
  and "assets"."isArchived" = $2
where
  "person"."ownerId" = $3
  and "asset_faces"."deletedAt" is null

-- PersonRepository.refreshFaces
with
  "added_embeddings" as (
    insert into
      "face_search" ("faceId", "embedding")
    values
      ($1, $2)
  )
select
from
  (
    select
      1
  ) as "dummy"

-- PersonRepository.getFacesByIds
select
  "asset_faces".*,
  (
    select
      to_json(obj)
    from
      (
        select
          "assets".*
        from
          "assets"
        where
          "assets"."id" = "asset_faces"."assetId"
      ) as obj
  ) as "asset",
  (
    select
      to_json(obj)
    from
      (
        select
          "person".*
        from
          "person"
        where
          "person"."id" = "asset_faces"."personId"
      ) as obj
  ) as "person"
from
  "asset_faces"
where
  "asset_faces"."assetId" in ($1)
  and "asset_faces"."personId" in ($2)
  and "asset_faces"."deletedAt" is null

-- PersonRepository.getRandomFace
select
  "asset_faces".*
from
  "asset_faces"
where
  "asset_faces"."personId" = $1
  and "asset_faces"."deletedAt" is null

-- PersonRepository.getLatestFaceDate
select
  max("asset_job_status"."facesRecognizedAt")::text as "latestDate"
from
  "asset_job_status"

-- PersonRepository.deleteAssetFace
delete from "asset_faces"
where
  "asset_faces"."id" = $1

-- PersonRepository.softDeleteAssetFaces
update "asset_faces"
set
  "deletedAt" = $1
where
  "asset_faces"."id" = $2
