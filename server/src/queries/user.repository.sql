-- NOTE: This file is auto generated by ./sql-generator

-- UserRepository.get
select
  "id",
  "name",
  "email",
  "profileImagePath",
  "profileChangedAt",
  "createdAt",
  "updatedAt",
  "deletedAt",
  "isAdmin",
  "status",
  "oauthId",
  "profileImagePath",
  "shouldChangePassword",
  "storageLabel",
  "quotaSizeInBytes",
  "quotaUsageInBytes",
  (
    select
      coalesce(json_agg(agg), '[]')
    from
      (
        select
          "user_metadata".*
        from
          "user_metadata"
        where
          "users"."id" = "user_metadata"."userId"
      ) as agg
  ) as "metadata"
from
  "users"
where
  "users"."id" = $1
  and "users"."deletedAt" is null

-- UserRepository.getAdmin
select
  "id",
  "name",
  "email",
  "profileImagePath",
  "profileChangedAt",
  "createdAt",
  "updatedAt",
  "deletedAt",
  "isAdmin",
  "status",
  "oauthId",
  "profileImagePath",
  "shouldChangePassword",
  "storageLabel",
  "quotaSizeInBytes",
  "quotaUsageInBytes"
from
  "users"
where
  "users"."isAdmin" = $1
  and "users"."deletedAt" is null

-- UserRepository.hasAdmin
select
  "users"."id"
from
  "users"
where
  "users"."isAdmin" = $1
  and "users"."deletedAt" is null

-- UserRepository.getByEmail
select
  "id",
  "name",
  "email",
  "profileImagePath",
  "profileChangedAt",
  "createdAt",
  "updatedAt",
  "deletedAt",
  "isAdmin",
  "status",
  "oauthId",
  "profileImagePath",
  "shouldChangePassword",
  "storageLabel",
  "quotaSizeInBytes",
  "quotaUsageInBytes"
from
  "users"
where
  "email" = $1
  and "users"."deletedAt" is null

-- UserRepository.getByStorageLabel
select
  "id",
  "name",
  "email",
  "profileImagePath",
  "profileChangedAt",
  "createdAt",
  "updatedAt",
  "deletedAt",
  "isAdmin",
  "status",
  "oauthId",
  "profileImagePath",
  "shouldChangePassword",
  "storageLabel",
  "quotaSizeInBytes",
  "quotaUsageInBytes"
from
  "users"
where
  "users"."storageLabel" = $1
  and "users"."deletedAt" is null

-- UserRepository.getByOAuthId
select
  "id",
  "name",
  "email",
  "profileImagePath",
  "profileChangedAt",
  "createdAt",
  "updatedAt",
  "deletedAt",
  "isAdmin",
  "status",
  "oauthId",
  "profileImagePath",
  "shouldChangePassword",
  "storageLabel",
  "quotaSizeInBytes",
  "quotaUsageInBytes"
from
  "users"
where
  "users"."oauthId" = $1
  and "users"."deletedAt" is null

-- UserRepository.getDeletedAfter
select
  "id"
from
  "users"
where
  "users"."deletedAt" < $1

-- UserRepository.getList (with deleted)
select
  "id",
  "name",
  "email",
  "profileImagePath",
  "profileChangedAt",
  "createdAt",
  "updatedAt",
  "deletedAt",
  "isAdmin",
  "status",
  "oauthId",
  "profileImagePath",
  "shouldChangePassword",
  "storageLabel",
  "quotaSizeInBytes",
  "quotaUsageInBytes",
  (
    select
      coalesce(json_agg(agg), '[]')
    from
      (
        select
          "user_metadata".*
        from
          "user_metadata"
        where
          "users"."id" = "user_metadata"."userId"
      ) as agg
  ) as "metadata"
from
  "users"
order by
  "createdAt" desc

-- UserRepository.getList (without deleted)
select
  "id",
  "name",
  "email",
  "profileImagePath",
  "profileChangedAt",
  "createdAt",
  "updatedAt",
  "deletedAt",
  "isAdmin",
  "status",
  "oauthId",
  "profileImagePath",
  "shouldChangePassword",
  "storageLabel",
  "quotaSizeInBytes",
  "quotaUsageInBytes",
  (
    select
      coalesce(json_agg(agg), '[]')
    from
      (
        select
          "user_metadata".*
        from
          "user_metadata"
        where
          "users"."id" = "user_metadata"."userId"
      ) as agg
  ) as "metadata"
from
  "users"
where
  "users"."deletedAt" is null
order by
  "createdAt" desc

-- UserRepository.getUserStats
select
  "users"."id" as "userId",
  "users"."name" as "userName",
  "users"."quotaSizeInBytes" as "quotaSizeInBytes",
  count(*) filter (
    where
      (
        "assets"."type" = $1
        and "assets"."isVisible" = $2
      )
  ) as "photos",
  count(*) filter (
    where
      (
        "assets"."type" = $3
        and "assets"."isVisible" = $4
      )
  ) as "videos",
  coalesce(
    sum("exif"."fileSizeInByte") filter (
      where
        "assets"."libraryId" is null
    ),
    0
  ) as "usage",
  coalesce(
    sum("exif"."fileSizeInByte") filter (
      where
        (
          "assets"."libraryId" is null
          and "assets"."type" = $5
        )
    ),
    0
  ) as "usagePhotos",
  coalesce(
    sum("exif"."fileSizeInByte") filter (
      where
        (
          "assets"."libraryId" is null
          and "assets"."type" = $6
        )
    ),
    0
  ) as "usageVideos"
from
  "users"
  left join "assets" on "assets"."ownerId" = "users"."id"
  left join "exif" on "exif"."assetId" = "assets"."id"
where
  "assets"."deletedAt" is null
group by
  "users"."id"
order by
  "users"."createdAt" asc

-- UserRepository.updateUsage
update "users"
set
  "quotaUsageInBytes" = "quotaUsageInBytes" + $1,
  "updatedAt" = $2
where
  "id" = $3::uuid
  and "users"."deletedAt" is null

-- UserRepository.syncUsage
update "users"
set
  "quotaUsageInBytes" = (
    select
      coalesce(sum("exif"."fileSizeInByte"), 0) as "usage"
    from
      "assets"
      left join "exif" on "exif"."assetId" = "assets"."id"
    where
      "assets"."libraryId" is null
      and "assets"."ownerId" = "users"."id"
  ),
  "updatedAt" = $1
where
  "users"."deletedAt" is null
  and "users"."id" = $2::uuid
